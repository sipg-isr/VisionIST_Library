"""colmap box - ground-truth SfM through the shared envelope.

Runs the real COLMAP (and its GLOMAP "global" mapper) on a set of images and
returns camera poses, 3D points and intrinsics. COLMAP is the reference
solver: slow but self-contained - no depth, no tracks, no calibration needed.

Design notes (see the box README for the user-facing contract):

* ``reconstruct`` (default)  - run one full COLMAP pipeline in this call.
  The gRPC reply comes at the END, which for a big set is minutes. That is
  fine: it is a ground-truth box, and the caller controls the size.
* ``log`` - the box cannot stream over the unary RPC, so it records every
  line of COLMAP's own output (plus box-level stage lines) in a bounded log
  and serves it on demand. A client polls ``log`` while the call is in
  flight and sees live progress - no ``docker logs`` access needed.
* ``reset`` - unlike most boxes it has teeth: it cancels the running
  reconstruction (SIGTERM; COLMAP writes partial results on SIGTERM).
* Because ``log``/``reset`` must answer while a reconstruction occupies a
  worker, the box ships ``max_workers: 2`` (one for the job, one to serve
  the lightweight calls) - see the limits in box.yaml.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402
import serve  # noqa: E402

import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = "PORT"

#: The config section this box answers to. Must equal `key` in box.yaml.
BOX_KEY = "colmap"

#: Where one job's working files live (database, images, sparse model).
JOBS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

#: How many lines of COLMAP output `log` remembers (and returns).
LOG_MAX_LINES = 20_000

DEFAULTS = {
    "mapper": "incremental",        # incremental | global (GLOMAP) | hierarchical
    "matcher": "auto",              # auto | exhaustive | sequential
    "sequential_overlap": 5,
    "max_image_size": 3200,
    "max_num_features": 8192,
    "use_gpu": "auto",              # auto | true | false
    "single_camera": True,
    "intrinsics": "",               # e.g. "PINHOLE 640 480 500 500 320 240"
    "max_runtime_s": 0,             # 0 = no budget; otherwise cancel at budget
    "tail": 100,                    # `log`: lines to return
}

#: COLMAP camera model -> number of parameters in cameras.txt.
CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "OPENCV": 8, "OPENCV_FISHEYE": 9, "FULL_OPENCV": 12, "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
    "RAD_TAN_THIN_PRISM_FISHEYE": 12, "SIMPLE_DIVISION": 3, "DIVISION": 4,
    "SIMPLE_FISHEYE": 3, "FISHEYE": 4, "EUCM": 5, "EQUIRECTANGULAR": 3,
    "SKEWED_PINHOLE": 5,
}


def _done(extra, data=None, encoding=None):
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _empty_request():
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "empty_request"}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: {"status": "error", "error": message}}))


#: COLMAP is on the image PATH (official colmap/colmap base image).
def _colmap_bin():
    return shutil.which("colmap") or "colmap"


def _gpu_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15)
        return out.returncode == 0 and "GPU" in (out.stdout or "")
    except Exception:                                  # noqa: BLE001
        return False


class Job:
    """One reconstruction: its log, its state, its running subprocess.

    There is at most one active job (the GPU is not divisible); `log` and
    `reset` are served from this object by other gRPC workers.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"          # idle | running | done | error | cancelled
        self.stage = ""
        self.started = None
        self.finished = None
        self.timenow = {"t": time.time()}
        self.log = deque(maxlen=LOG_MAX_LINES)
        self.error = None
        self.timed_out = False
        self.result = None           # (config_extra, data, encoding) on success
        self.proc = None
        self._proc_thread = None
        self._stopped = threading.Event()   # worker released the proc

    def log_line(self, line):
        with self.lock:
            self.log.append(line)
        logging.info("[colmap] %s", line)

    def tail(self, n):
        with self.lock:
            return list(self.log)[-max(1, int(n)):]

    def state_snapshot(self):
        with self.lock:
            return {
                "running": self.state == "running",
                "state": self.state,
                "stage": self.stage,
                "started_s": (self.started is not None),
                "elapsed_s": round((time.time() - self.started), 1)
                if self.started else None,
                "lines": len(self.log),
            }

    def cancel(self):
        """SIGTERM (COLMAP writes partial results), then SIGKILL if it lingers."""
        with self.lock:
            proc = self.proc
            if proc is None or self.state not in ("running",):
                return self.state
            self.state = "cancelled"
        if proc is not None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if self._proc_thread and self._proc_thread.is_alive():
                self._proc_thread.join(timeout=15)


#: Singleton shared by every gRPC worker thread.
JOB = Job()


def _np_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def _run_stage(job, stage, argv, deadline):
    """Run one COLMAP stage, streaming its output into the job log.

    Returns False if the stage did not finish cleanly (caller decides
    whether that is fatal or a partial result)."""
    job.log_line(f"[box] stage: {stage}")
    job.stage = stage
    cmd = [_colmap_bin()] + list(argv)
    job.log_line("[box] $ " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with job.lock:
        job.proc = proc

    def stream():
        try:
            for raw in proc.stdout:
                job.log_line(raw.rstrip("\n"))
        finally:
            with job.lock:
                job._stopped.set()

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()

    while True:
        rc = proc.poll()
        if rc is not None:
            thread.join(timeout=10)
            break
        if deadline is not None and time.time() >= deadline:
            job.timed_out = True
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            thread.join(timeout=10)
            job.log_line("[box] budget reached: cancelling")
            return False
        time.sleep(0.4)

    if rc != 0:
        # COLMAP prints the error as its last lines; surface it, not a raw rc.
        job.log_line(f"[box] stage {stage!r} exited with {rc}")
        return False
    job.log_line(f"[box] stage {stage!r} done in {time.time() - t0:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Reading COLMAP's text model (cameras.txt / images.txt / points3D.txt)
# ---------------------------------------------------------------------------

def _qvec_to_rot(qvec):
    w, x, y, z = (float(v) for v in qvec)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _parse_model(model_dir):
    """cameras.txt + images.txt + points3D.txt -> numpy outputs + metadata.

    Returns (cameras, points, points_rgb, intrinsics, cam_models,
             reg_names, unreg_count)."""
    def tokens(path):
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                yield line.split()

    cameras = {}
    cam_models = {}
    for t in tokens(os.path.join(model_dir, "cameras.txt")):
        cid = int(float(t[0]))
        model = t[1]
        w, h = int(float(t[2])), int(float(t[3]))
        n = CAMERA_MODEL_PARAMS.get(model)
        if n is None:
            raise ValueError(f"unknown COLMAP camera model {model!r}")
        params = [float(v) for v in t[4:4 + n]]
        cameras[cid] = (w, h, params)
        cam_models[cid] = model

    reg_names, poses, cam_ids = [], [], []
    unregistered = 0
    toks = list(tokens(os.path.join(model_dir, "images.txt")))
    i = 0
    while i < len(toks):
        t = toks[i]
        if len(t) == 9 and t[1] == "-1":          # single line, unregistered
            unregistered += 1
            i += 1
            continue
        # registered: pose line, then a (possibly empty) points2D line
        qvec = [float(v) for v in t[1:5]]
        tvec = [float(v) for v in t[5:8]]
        cid = int(float(t[8]))
        name = t[9] if len(t) > 9 else f"image{int(float(t[0]))}"
        R = _qvec_to_rot(qvec)
        poses.append(np.hstack([R, np.array(tvec)[:, None]]))
        cam_ids.append(cid)
        reg_names.append(name)
        i += 2

    points_list, rgb_list = [], []
    for t in tokens(os.path.join(model_dir, "points3D.txt")):
        points_list.append([float(v) for v in t[1:4]])
        if len(t) >= 6:
            rgb_list.append([max(0, min(255, int(float(v)))) for v in t[4:7]])
        else:
            rgb_list.append([0, 0, 0])

    cameras_arr = np.stack(poses) if poses else np.zeros((0, 3, 4))
    points_arr = (np.array(points_list, dtype=np.float64)
                  if points_list else np.zeros((0, 3)))
    intrinsics = np.zeros((len(cam_ids), 3, 3))
    one_focal = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_FISHEYE"}
    for k, cid in enumerate(cam_ids):
        _, _, params = cameras[cid]
        model = cam_models[cid]
        if model in one_focal and len(params) >= 3:
            fx = fy = float(params[0])            # (f, cx, cy[, k...])
            cx, cy = (float(params[1]), float(params[2]))
        elif len(params) >= 4:
            fx, fy, cx, cy = (float(v) for v in params[:4])
        else:
            w, h = cameras[cid][0], cameras[cid][1]
            fx = fy = float(params[0]) if params else w
            cx, cy = float(params[1]) if len(params) > 1 else w / 2, h / 2
        intrinsics[k] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return (cameras_arr, points_arr, np.array(rgb_list), intrinsics,
            {k: cam_models[k] for k in sorted(cam_models)}, reg_names,
            unregistered)


def _reproj_stats(model_dir):
    """Mean reprojection error via `colmap model_analyzer` (4.x uses --path)."""
    try:
        out = subprocess.run(
            [_colmap_bin(), "model_analyzer", "--path", model_dir],
            capture_output=True, text=True, timeout=120)
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"Mean reprojection error:\s*([\d.eE+-]+)\s*px", text)
        if m:
            return float(m.group(1))
    except Exception:                              # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

def _param(section, name, params):
    value = params.get(name, section.get(name, DEFAULTS[name]))
    return value


class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):

    def Process(self, request, context):
        start = time.time()
        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or \
                    not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")
            section = config[BOX_KEY]
            command = section.get("command") or "reconstruct"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            if command == "reset":
                state = JOB.cancel()
                return _done({"action": "reset", "state_before": state})
            if command == "log":
                tail = int(_param(section, "tail", parameters))
                lines = JOB.tail(tail)
                return _done(JOB.state_snapshot(),
                             data={"log": wrap_value("".join(l + "\n"
                                                              for l in lines).encode())},
                             encoding={"log": "identity"})
            if command != "reconstruct":
                return _error(f"unknown command {command!r} "
                              f"(expected 'reconstruct', 'log' or 'reset')")

            images = unwrap_value(request.data["images"]) \
                if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [bytes(images)]
            if not images:
                return _empty_request()

            images = [bytes(im) for im in images]

            names = None
            if "image_names" in request.data:
                names = unwrap_value(request.data["image_names"])
                if isinstance(names, str):
                    names = [names]
                names = [str(n) for n in names]
                if len(names) != len(images):
                    return _error(f"image_names has {len(names)} entries, "
                                  f"images has {len(images)}")

            return self._reconstruct(images, names, section, parameters, start)
        except Exception as e:                        # noqa: BLE001
            logging.exception("Error in Process")
            return _error(str(e))

    # -- reconstruct -------------------------------------------------------

    def _reconstruct(self, images, names, section, parameters, start):
        if JOB.state_snapshot()["running"]:
            return _error("a reconstruction is already running; "
                          "poll command='log' for progress, or command='reset'")

        try:
            mapper = str(_param(section, "mapper", parameters)).lower()
            if mapper not in ("incremental", "global", "hierarchical"):
                return _error(f"mapper must be incremental|global|hierarchical, "
                              f"got {mapper!r}")
            matcher = str(_param(section, "matcher", parameters)).lower()
            if matcher not in ("auto", "exhaustive", "sequential"):
                return _error(f"matcher must be auto|exhaustive|sequential, "
                              f"got {matcher!r}")
            if matcher == "auto" and names:
                matcher = "sequential"
            max_runtime_s = float(_param(section, "max_runtime_s", parameters))
            use_gpu_raw = str(_param(section, "use_gpu", parameters)).lower()
            if use_gpu_raw in ("auto", ""):
                gpu = _gpu_available()
            else:
                gpu = use_gpu_raw in ("true", "1", "yes", "on")
                if use_gpu_raw not in ("false", "0", "no", "off"):
                    return _error(f"use_gpu must be auto|true|false, "
                                  f"got {use_gpu_raw!r}")
            single_camera = bool(_param(section, "single_camera", parameters))
            intrinsics = str(_param(section, "intrinsics", parameters)).strip()
        except Exception as e:                        # noqa: BLE001
            return _error(f"bad parameters: {e}")

        workdir = os.path.join(JOBS_ROOT, f"job_{int(start * 1000)}")
        images_dir = os.path.join(workdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i, raw in enumerate(images):
            fname = names[i] if names else f"frame_{i:04d}.jpg"
            with open(os.path.join(images_dir, os.path.basename(fname)), "wb") as fh:
                fh.write(raw)

        if intrinsics:
            parts = intrinsics.split()
            cam_model = parts[0].upper()
            rest = " ".join(str(v) for v in parts[1:])
            sift_extra = ["--ImageReader.single_camera",
                          "1" if single_camera else "0",
                          "--ImageReader.camera_model", cam_model,
                          "--ImageReader.camera_params", rest]
        else:
            sift_extra = ["--ImageReader.single_camera",
                          "1" if single_camera else "0"]

        def sift(*extra):
            argv = ["feature_extractor",
                    "--database_path", os.path.join(workdir, "database.db"),
                    "--image_path", images_dir,
                    "--FeatureExtraction.max_image_size",
                    str(int(_param(section, "max_image_size", parameters))),
                    "--SiftExtraction.max_num_features",
                    str(int(_param(section, "max_num_features", parameters)))]
            argv += ["--FeatureExtraction.use_gpu", "1" if gpu else "0"]
            return argv + list(extra)

        steps = [["feature_extraction", sift(*sift_extra)]]
        if matcher == "sequential":
            overlap = int(_param(section, "sequential_overlap", parameters))
            steps.append(["matching", ["sequential_matcher",
                                       "--database_path",
                                       os.path.join(workdir, "database.db"),
                                       "--SequentialMatching.overlap", str(overlap),
                                       "--FeatureMatching.use_gpu", "1" if gpu else "0"]])
        else:
            steps.append(["matching", ["exhaustive_matcher",
                                       "--database_path",
                                       os.path.join(workdir, "database.db"),
                                       "--FeatureMatching.use_gpu", "1" if gpu else "0"]])

        if mapper == "global":
            # GLOMAP runs on the same database (it only reads); no copy needed.
            db_path = os.path.join(workdir, "database.db")
            steps.append(["view_graph_calibration",
                          ["view_graph_calibrator",
                           "--database_path", db_path]])
            steps.append(["global_mapping",
                          ["global_mapper",
                           "--database_path", db_path,
                           "--image_path", images_dir,
                           "--output_path", os.path.join(workdir, "sparse")]])
        else:
            cmd = "hierarchical_mapper" if mapper == "hierarchical" else "mapper"
            steps.append([f"{mapper}_mapping",
                          [cmd,
                           "--database_path", os.path.join(workdir, "database.db"),
                           "--image_path", images_dir,
                           "--output_path", os.path.join(workdir, "sparse")]])

        # Bundle adjustment runs with COLMAP's default backend (Ceres); the
        # experimental Caspar GPU backend stays off deliberately.

        deadline = time.time() + max_runtime_s if max_runtime_s > 0 else None

        # COLMAP 4.x `mapper`/`global_mapper` want `--output_path` to exist
        # as a directory; the model lands in `output_path/0/`.
        os.makedirs(os.path.join(workdir, "sparse"), exist_ok=True)

        JOB.state = "running"
        JOB.started = start
        JOB.error = None
        JOB.timed_out = False
        JOB.result = None
        JOB.log_line(f"[box] reconstruct: {len(images)} images, "
                     f"mapper={mapper}, matcher={matcher}, gpu={gpu}")

        ok_all = True
        stage_times = {}
        for stage, argv in steps:
            t0 = time.time()
            ok = _run_stage(JOB, stage, argv, deadline)
            stage_times[stage] = round(time.time() - t0, 2)
            if not ok and not JOB.timed_out:
                ok_all = False
                JOB.state = "error"
                JOB.finished = time.time()
                JOB.stage = stage
                break
            if JOB.timed_out:
                break     # budget is spent - don't queue further stages; try collection
        timed_out = JOB.timed_out

        JOB.stage = "collect"
        model_dir = self._find_model(workdir)
        if model_dir and not os.path.isfile(os.path.join(model_dir, "images.txt")):
            # COLMAP 4.x writes binary models (.bin) - convert to text so the
            # parsers (and users) see the classic, stable format.
            txt_dir = os.path.join(workdir, "model_txt")
            os.makedirs(txt_dir, exist_ok=True)
            if _run_stage(JOB, "model_conversion",
                          ["model_converter", "--input_path", model_dir,
                           "--output_path", txt_dir, "--output_type", "TXT"],
                          deadline) \
                    and os.path.isfile(os.path.join(txt_dir, "images.txt")):
                model_dir = txt_dir
        if model_dir:
            try:
                (cameras_arr, points_arr, rgb, intrinsics, cam_models,
                 reg_names, unreg) = _parse_model(model_dir)
                reproj = _reproj_stats(model_dir)

                # Map registered image names back to the request's image order.
                by_name = {os.path.basename(n): k
                           for k, n in enumerate(names or [
                               f"frame_{i:04d}.jpg" for i in range(len(images))])}
                frame_ids = [by_name.get(os.path.basename(n), -1)
                             for n in reg_names]

                data = {
                    "cameras": wrap_value(_np_bytes(cameras_arr)),      # (F,3,4)
                    "points": wrap_value(_np_bytes(points_arr)),        # (P,3)
                    "point_colors": wrap_value(_np_bytes(rgb)),         # (P,3)
                    "intrinsics": wrap_value(_np_bytes(intrinsics)),    # (F,3,3)
                    "frame_ids": wrap_value(
                        _np_bytes(np.array(frame_ids, dtype=np.int64))),
                    "image_names": wrap_value(
                        _np_bytes(np.array(reg_names))),
                }
                extra = {
                    "mapper": mapper, "matcher": matcher, "gpu": gpu,
                    "num_images": len(images),
                    "num_registered": len(reg_names),
                    "num_unregistered": unreg,
                    "num_points": len(points_arr),
                    "reproj_error": reproj,
                    "stage_times_s": stage_times,
                    "partial": bool(timed_out),
                    "runtime": round(time.time() - start, 2),
                }
                encoding = {
                    "cameras": "numpy", "points": "numpy",
                    "point_colors": "numpy", "intrinsics": "numpy",
                    "frame_ids": "numpy", "image_names": "numpy",
                }
            except Exception as e:                    # noqa: BLE001
                JOB.state = "error"
                JOB.error = f"model collection failed: {e}"
                JOB.finished = time.time()
                JOB.log_line(f"[box] {JOB.error}")
                return _error(JOB.error)
            JOB.state = "done"
            JOB.result = (extra, data, encoding)
        else:
            JOB.state = "error"
            JOB.error = ("no sparse model produced"
                         + (" (budget reached before a model could be written)"
                            if timed_out else ""))
            JOB.finished = time.time()
            JOB.log_line(f"[box] {JOB.error}")
            return _error(JOB.error)

        JOB.finished = time.time()
        return _done(JOB.result[0], data=JOB.result[1], encoding=JOB.result[2])

    @staticmethod
    def _find_model(workdir):
        sparse = os.path.join(workdir, "sparse")
        if os.path.isdir(sparse):
            candidates = sorted(
                (d for d in os.listdir(sparse)
                 if os.path.isdir(os.path.join(sparse, d))),
                key=lambda d: int(d) if d.isdigit() else 0)
            for d in reversed(candidates):
                full = os.path.join(sparse, d)
                if os.path.isfile(os.path.join(full, "images.txt")) or \
                        os.path.isfile(os.path.join(full, "images.bin")):
                    return full
        return None


def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        return port if port > 0 else None
    except ValueError:
        logging.exception("Invalid port value")
        return None


def run_server(server):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        JOB.cancel()
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options(),
        maximum_concurrent_rpcs=serve.max_concurrent_rpcs())
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
