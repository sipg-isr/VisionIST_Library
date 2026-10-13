"""
MoGe-3 Box: Monocular Geometry Estimation Service

Estimates 3D geometry (point maps, depth maps, normal maps, intrinsics) from
single images using Microsoft's MoGe-3 model.

GPU lifecycle:
- Model loads on CPU at startup (fast start, no VRAM)
- Auto-moves to GPU on first request (in-place .to() on torch modules)
- Falls back to CPU after ~60s idle (releases cached features)
"""

import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import json
import logging
import os
import pickle
import threading
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import sys
sys.path.append('./protos')
import pipeline_pb2
import pipeline_pb2_grpc
from aux import wrap_value, unwrap_value
import serve

_IDLE_TIMEOUT = 60  # seconds before GPU→CPU fallback

# MoGe-3 model versions
MODEL_V3_GIGANTIC = "Ruicheng/moge-3-vitg"  # 1.25B params, best quality
MODEL_V3_LARGE = "Ruicheng/moge-3-vitl"      # 370M params, faster
MODEL_V3_BASE = "moge-3-vitl"  # Current default


class MoGeBox(pipeline_pb2_grpc.PipelineServiceServicer):
    def __init__(self, model_name: str = MODEL_V3_LARGE):
        """Initialize MoGe-3 service with CPU at startup."""
        self._model = None
        self._device = "cpu"
        self._model_name = model_name
        self._last_request_time = time.time()
        self._lock = threading.Lock()
        self._model_loaded = threading.Event()
        
        # GPU memory optimization before model load
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        
        # Load model asynchronously (don't block startup)
        threading.Thread(target=self._load_model_async, daemon=True).start()
        
        # Watchdog for GPU→CPU fallback
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        
        logging.info(f"MoGe-3 service initialized (model: {model_name}, device: {self._device})")

    def _load_model_async(self):
        """Load MoGe-3 model in background thread."""
        try:
            from moge.model.v3 import MoGeModel
            
            logging.info(f"Loading MoGe-3 model from {self._model_name}...")
            self._model = MoGeModel.from_pretrained(self._model_name)
            
            with self._lock:
                # Model stays on CPU initially (device set in Process())
                torch.cuda.empty_cache()
                self._model_loaded.set()
            
            logging.info(f"MoGe-3 model loaded successfully")
        except Exception as e:
            logging.exception(f"Failed to load MoGe-3 model: {e}")
            self._model_loaded.set()  # Signal that load is done (even if failed)

    def _ensure_model_ready(self) -> bool:
        """Wait for model to be loaded. Returns False if model failed to load."""
        if self._model is None:
            if not self._model_loaded.wait(timeout=120):  # 2 min timeout
                logging.error("Model load timed out")
                return False
            if self._model is None:
                logging.error("Model failed to load")
                return False
        return True

    def _move_to_device(self, target: str):
        """Move model to target device (in-place, no reload)."""
        if self._model is None:
            return
        
        self._model.to(torch.device(target))
        self._device = target
        logging.info(f"MoGe-3 model moved to {target}")
        
        if target.startswith("cuda"):
            torch.cuda.empty_cache()
        else:
            # .to("cpu") only relocates registered params/buffers; MoGe keeps
            # extra state (feature caches / intermediate buffers) as plain
            # attributes that would keep pinning VRAM.  Null them, then give
            # the allocator blocks back to the driver.
            self._release_cuda_leftovers()
            torch.cuda.empty_cache()

    def _release_cuda_leftovers(self) -> None:
        """Drop non-parameter CUDA tensors left on the model after parking.

        Walks the model object graph and nulls any attribute that is (or
        holds a dict/list of) a CUDA tensor.  Safe to run only when parking:
        registered parameters/buffers are already on CPU at that point, and
        any remaining CUDA tensors are by definition caches/intermediates
        (they are recomputed on the next inference).
        """
        if self._model is None:
            return
        seen: set = set()
        cleared = []

        def _purge(obj):
            if obj is None or id(obj) in seen:
                return
            try:
                attrs = vars(obj)
            except TypeError:
                return
            seen.add(id(obj))
            for name, val in list(attrs.items()):
                try:
                    if torch.is_tensor(val):
                        if val.is_cuda:
                            setattr(obj, name, None)
                            cleared.append(name)
                    elif isinstance(val, dict):
                        if any(torch.is_tensor(v) and v.is_cuda for v in val.values()):
                            val.clear()
                            cleared.append(name)
                        else:
                            _purge(val)
                    elif isinstance(val, (list, tuple)):
                        if any(torch.is_tensor(v) and v.is_cuda for v in val):
                            setattr(obj, name, [] if isinstance(val, list) else tuple())
                            cleared.append(name)
                        else:
                            _purge(val)
                except Exception:
                    continue

        for module in self._model.modules():
            _purge(module)
        if cleared:
            logging.info(f"MoGe-3 parking: released {len(cleared)} CUDA-cached attribute(s): {cleared[:10]}")

    def _watchdog_loop(self):
        """Fallback to CPU after IDLE_TIMEOUT seconds of inactivity."""
        while True:
            time.sleep(10)
            with self._lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda"):
                    self._move_to_device("cpu")
                    logging.info(f"MoGe-3 fallback to CPU after {idle_time:.1f}s idle")

    def Process(self, request, context):
        """Process gRPC Envelope request and return Envelope response."""
        try:
            # Parse config
            cfg = json.loads(request.config_json)
            box_cfg = cfg.get("moge", {})
            
            # Reset command (always accepted)
            if box_cfg.get("command") == "reset":
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "done",
                            "action": "reset"
                        }
                    })
                )
            
            # Check for images in request
            if "images" not in request.data:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "empty_request",
                            "error": "No images provided in data"
                        }
                    })
                )
            
            images = unwrap_value(request.data["images"])
            # A lone image arrives as a bare bytes value (Value.b), not a
            # one-element BytesList; iterating it would yield ints.
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images or len(images) == 0:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "empty_request",
                            "error": "Empty images list"
                        }
                    })
                )
            if not all(isinstance(b, (bytes, bytearray)) for b in images):
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "error",
                            "error": "data.images must be image file bytes "
                                     "(send the file contents, e.g. "
                                     "pathlib.Path(...), not a path string)"
                        }
                    })
                )
            
            # Get parameters with safe defaults
            parameters = box_cfg.get("parameters", {}) or {}
            
            # Parse optional FOV
            fov_x = parameters.get("fov_x")  # degrees, optional
            
            # Inference parameters
            refine_steps = parameters.get("refine_steps", 3)  # default 3 steps for v3
            resolution_level = parameters.get("resolution_level", 9)  # 0-9
            fp16 = parameters.get("fp16", True)  # default True for speed
            
            # Device selection (explicit params.device wins, else auto-GPU)
            target_device = parameters.get("device")

            # MoGe-3's sparse volumetric refinement runs CUDA-only Triton
            # kernels (flex_gemm); CPU inference is not supported. Fail fast
            # with a clean error instead of a CUDA stack trace.
            if target_device is not None and torch.device(target_device).type == "cpu":
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "error",
                            "error": "CPU inference is not supported for MoGe-3 "
                                     "(CUDA-only sparse refinement kernels). "
                                     "Use the default auto-GPU path."
                        }
                    })
                )
            
            start_time = time.time()
            
            # Wait for model to be ready
            if not self._ensure_model_ready():
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({
                        "moge": {
                            "status": "error",
                            "error": "Model not loaded"
                        }
                    })
                )
            
            # Auto-move to GPU if available
            with self._lock:
                self._last_request_time = time.time()
                if not target_device:
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                if target_device != self._device:
                    self._move_to_device(target_device)
            
            # Process each image
            results = []
            for img_bytes in images:
                result = self._infer_single(
                    img_bytes, 
                    fov_x=fov_x,
                    refine_steps=refine_steps,
                    resolution_level=resolution_level,
                    fp16=fp16
                )
                results.append(result)
            
            elapsed = time.time() - start_time
            
            # Pack results (zstd + pickle for heavy tensor output)
            import zstandard as zstd
            blob = zstd.ZstdCompressor().compress(pickle.dumps(results))
            
            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "moge": {
                        "status": "done",
                        "num_images": len(images),
                        "runtime": elapsed,
                        "device": self._device,
                        # Declare encoding for automatic decoding
                        "encoding": "zstd_pickle"
                    }
                }),
                data={"results": wrap_value(blob)}
            )
            
        except Exception as e:
            logging.exception("Inference failed")
            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "moge": {
                        "status": "error",
                        "error": str(e)
                    }
                })
            )

    def _infer_single(
        self, 
        image_bytes: bytes,
        fov_x: Optional[float] = None,
        refine_steps: int = 3,
        resolution_level: int = 9,
        fp16: bool = True
    ) -> Dict[str, Any]:
        """
        Run MoGe-3 inference on a single image.
        
        Args:
            image_bytes: JPEG/PNG image as bytes
            fov_x: Optional horizontal field of view in degrees
            refine_steps: Number of sparse refinement steps (MoGe-3 only)
            resolution_level: Inference resolution level [0-9]
            fp16: Use FP16 for faster inference
            
        Returns:
            Dictionary with:
                - points: (H, W, 3) metric point map [OpenCV camera coords]
                - depth: (H, W) metric depth map
                - intrinsics: (3, 3) normalized camera intrinsics
                - mask: (H, W) binary valid mask
                - normal: (H, W, 3) normal map [OpenCV camera coords]
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image")
        
        # Convert to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(
            img_rgb / 255.0, 
            dtype=torch.float32, 
            device=torch.device(self._device)
        ).permute(2, 0, 1)  # (3, H, W)
        
        # Prepare inputs for MoGe-3
        input_dict = {
            "image": img_tensor,
            "refine_steps": refine_steps,
        }
        
        if fov_x is not None:
            input_dict["fov_x"] = fov_x
        
        # Set precision
        if fp16 and self._device.startswith("cuda"):
            # Enable FP16 with autocast
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with self._lock:
                    output = self._model.infer(
                        input_dict["image"],
                        fov_x=input_dict.get("fov_x"),
                        refine_steps=refine_steps
                    )
        else:
            with self._lock:
                output = self._model.infer(
                    input_dict["image"],
                    fov_x=input_dict.get("fov_x"),
                    refine_steps=refine_steps
                )
        
        # Extract outputs (already on correct device)
        result = {
            "points": output["points"].detach().cpu().numpy(),
            "depth": output["depth"].detach().cpu().numpy(),
            "intrinsics": output["intrinsics"].detach().cpu().numpy(),
            "mask": output["mask"].detach().cpu().numpy(),
            "normal": output["normal"].detach().cpu().numpy(),
        }
        
        # Clear GPU cache periodically
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        return result


def run_server():
    """Start gRPC server."""
    port = int(os.getenv("PORT", 8061))
    
    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers(10)),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    
    # Default to MoGe-3 large model (faster); use gigantic for best quality
    box = MoGeBox(model_name=MODEL_V3_LARGE)
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(box, server)
    
    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)
    
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"MoGe-3 box started on {target} (model: Ruicheng/moge-3-vitl)")
    server.wait_for_termination()


if __name__ == "__main__":
    run_server()
