"""Every task, driven over real gRPC against real YOLO26 checkpoints.

    python3 test/test_yolo_tasks.py          # starts its own server in-process

Needs the box's own dependencies (ultralytics, torch, lap) plus the
checkpoints, which ultralytics downloads on first use: yolo26n.pt and the
-pose / -seg / -obb / -cls variants, about 25 MB in total. No GPU required.

It checks what the earlier detect-only tests could not: that each command
produces the fields it promises, that pose keypoints and segmentation
polygons land inside the frame (the trap being that ultralytics' raw
masks.data is at the model's input resolution, not the frame's), that
asking for a task the active checkpoint cannot do is refused with the name
of the checkpoint to use, and that classify comes back without track ids
because ultralytics refuses to track it.
"""
import os, sys, json, io, warnings, concurrent.futures as futures
warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_WEIGHTS", "yolo26n.pt")
os.environ["MAX_MESSAGE_MB"] = "256"
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "protos"), str(ROOT / "src")]
import numpy as np, cv2, grpc, pipeline_pb2, pipeline_pb2_grpc
from aux import wrap_value, unwrap_value
from yolo_service import PipelineService

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'ok  ' if cond else 'FAIL'} {name:46s} {detail}")

server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                     options=[("grpc.max_send_message_length", -1),
                              ("grpc.max_receive_message_length", -1)])
pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)
port = server.add_insecure_port("127.0.0.1:0"); server.start()
stub = pipeline_pb2_grpc.PipelineServiceStub(grpc.insecure_channel(
    f"127.0.0.1:{port}", options=[("grpc.max_send_message_length", -1),
                                  ("grpc.max_receive_message_length", -1)]))
# ultralytics ships a sample image with people and a bus in it.
_asset = __import__("ultralytics").__path__[0] + "/assets/bus.jpg"
img = open(_asset, "rb").read()

def call(section, data=None, timeout=300):
    env = pipeline_pb2.Envelope(config_json=json.dumps({"yolo": section}))
    for k,v in (data or {}).items(): env.data[k].CopyFrom(wrap_value(v))
    rep = stub.Process(env, timeout=timeout)
    cfg = json.loads(rep.config_json)["yolo"]
    fields = {k: unwrap_value(v) for k,v in rep.data.items()}
    det = json.loads(fields["detections"].decode()) if "detections" in fields else None
    return cfg, fields, det

sec = lambda cmd, **p: {"command": cmd, "session_id": "t1",
                        "parameters": {"save_annotated": False, **p}}

# ---- detect: unchanged for existing callers --------------------------------
cfg, f, det = call(sec("detect"), {"images":[img]})
r = det[0]
check("detect still works", cfg["status"]=="done" and len(r["boxes"])>0,
      f"{len(r['boxes'])} boxes, labels {r['labels'][:3]}")
check("detect record keeps its old keys",
      all(k in r for k in ("frame_index","width","height","boxes","class_ids","labels","scores","track_id")))
check("detect carries track ids", r["track_id"] is not None, str(r["track_id"]))

# ---- pose ------------------------------------------------------------------
cfg, f, det = call(sec("pose"), {"images":[img]})
r = det[0]
kp = np.asarray(r["keypoints"]) if r["keypoints"] else np.zeros((0,))
check("pose returns keypoints", cfg["status"]=="done" and kp.ndim==3 and kp.shape[1]==17 and kp.shape[2]==3,
      f"shape {kp.shape}")
check("pose names the COCO-17 joints", r["keypoint_names"] and r["keypoint_names"][0]=="nose",
      str(r["keypoint_names"][:3] if r["keypoint_names"] else None))
check("pose keypoints are inside the frame",
      kp.size>0 and kp[...,0].max()<=r["width"] and kp[...,1].max()<=r["height"],
      f"x<= {kp[...,0].max():.0f}/{r['width']}  y<= {kp[...,1].max():.0f}/{r['height']}")
check("pose tracks people too", r["track_id"] is not None, str(r["track_id"]))

# ---- segment ---------------------------------------------------------------
cfg, f, det = call(sec("segment"), {"images":[img]})
r = det[0]
polys = r["polygons"] or []
check("segment returns polygons", cfg["status"]=="done" and len(polys)>0,
      f"{len(polys)} contours, sizes {[len(p) for p in polys][:4]}")
allpts = np.concatenate([np.asarray(p) for p in polys]) if polys else np.zeros((0,2))
check("polygons are in frame pixels",
      allpts[:,0].max()<=r["width"] and allpts[:,1].max()<=r["height"],
      f"x<= {allpts[:,0].max():.0f}/{r['width']}  y<= {allpts[:,1].max():.0f}/{r['height']}")
check("segment keeps boxes and track ids", len(r["boxes"])>0 and r["track_id"] is not None)
json_sz = len(json.dumps(det).encode())
check("polygon payload is small", json_sz < 400_000, f"{json_sz/1024:.0f} KB for the whole reply")

# ---- segment with full masks ----------------------------------------------
cfg, f, det = call(sec("segment", return_masks=True), {"images":[img]})
masks = f.get("masks")
arr = np.load(io.BytesIO(masks[0]), allow_pickle=False) if masks else None
check("return_masks gives a frame-sized stack",
      arr is not None and arr.ndim==3 and (arr.shape[1],arr.shape[2])==(det[0]["height"],det[0]["width"]),
      f"{None if arr is None else arr.shape} vs frame {det[0]['height']}x{det[0]['width']}")
check("masks declared with the numpy codec", (cfg.get("encoding") or {}).get("masks")=="numpy",
      str(cfg.get("encoding")))

# ---- obb / classify --------------------------------------------------------
cfg, f, det = call(sec("obb"), {"images":[img]})
check("obb runs and reports its own container", cfg["status"]=="done" and "obb" in det[0],
      f"obb={det[0]['obb'] if det[0]['obb'] else 'no detections (DOTA model on a street photo)'}" [:60])
cfg, f, det = call(sec("classify"), {"images":[img]})
cl = det[0]["classification"]
check("classify returns a top-5", cfg["status"]=="done" and cl and len(cl["top5"])==5,
      f"top1={cl['top1']['label']!r} {cl['top1']['score']}" if cl else "")
check("classify has no boxes and no track ids",
      cl is not None and det[0]["boxes"]==[] and det[0]["track_id"] is None)

# ---- the guards ------------------------------------------------------------
cfg, _, _ = call({"command":"pose","session_id":"t1",
                  "parameters":{"weights":"yolo26n.pt","save_annotated":False}}, {"images":[img]})
check("wrong checkpoint for the task is caught",
      cfg["status"]=="error" and "is a detect model" in cfg["error"], cfg.get("error","")[:70])
cfg, _, _ = call(sec("frobnicate"), {"images":[img]})
check("unknown command rejected", cfg["status"]=="error", cfg.get("error","")[:50])
cfg, _, _ = call({"command":"reset","session_id":"t1"})
check("reset still works", cfg["status"]=="done" and cfg.get("action")=="reset")
cfg, f, det = call(sec("detect", nms=False), {"images":[img]})
check("nms=False (end-to-end head) accepted", cfg["status"]=="done",
      f"{len(det[0]['boxes'])} boxes" if det else cfg.get("error","")[:50])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL: print("failed:", ", ".join(FAIL))
server.stop(0)
sys.exit(1 if FAIL else 0)
