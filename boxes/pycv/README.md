# pycv

Runs the Python you send it, against OpenCV, and hands back whatever it
produces. A script or a one-liner in; named arrays, images and JSON out.

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:9075")
res = b.run(data={"code": "cv2.Canny(cv2.cvtColor(vs.image, cv2.COLOR_BGR2GRAY), 100, 200)",
                  "images": [pathlib.Path("dog.jpg")]},
            config={"pycv": {"command": "eval"}})
edges = res.result            # a decoded numpy array
```

## Read this first

**This box executes code it is sent.** Anyone who can reach its port can run
anything the container's user can run. That is the box's whole purpose, not a
flaw in it, but it makes `pycv` categorically different from every other box
in this registry: it belongs on a trusted network and its port should never be
published to one you do not control.

What the box *does* do is bound the damage of an accident — a runaway loop, a
40 GB allocation, a fork bomb, a segfault. Every request runs in a subprocess,
in a fresh directory that is deleted afterwards, in its own process group, under
a wall-clock timeout and `RLIMIT_AS` / `RLIMIT_CPU` / `RLIMIT_FSIZE`. A script
that hangs or spawns children is killed with its whole group; a script that
crashes cannot take the box down for other callers.

Container-level settings are the other half, and they belong in your compose
file rather than in the image:

```yaml
  pycv:
    image: docker.io/sipgisr/visionist-pycv:latest
    read_only: true
    tmpfs: [/tmp]
    cap_drop: [ALL]
    security_opt: [no-new-privileges:true]
    mem_limit: 4g
    pids_limit: 256
```

Blocking outbound traffic is deliberately *not* claimed here. Docker cannot
give a container published ports and no egress at the same time; if you need
that, run the fleet on an `internal: true` network with no published ports, or
block it at the host firewall.

## The two ways to send code

### `eval` — one string

```python
config={"pycv": {"command": "eval"}}
data={"code": "cv2.GaussianBlur(vs.image, (5, 5), 1.5)", "images": [img]}
```

The string is compiled as an expression; if that fails it is compiled as
statements, so a few lines work too. An expression's value comes back as
`result`. A **tuple is unpacked** into `result_0`, `result_1`, … because so
much of the cv2 API returns one:

```python
"cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)"   # -> result_0, result_1
```

Inside the string, `vs`, `cv2`, `np`, `images`, `image`, `texts`, `numbers`,
`args`, `inputs` and `emit` are all in scope.

### `run` — a set of files

```python
config={"pycv": {"command": "run"}}
data={"code": [main_bytes, helper_bytes], "names": ["main.py", "helper.py"]}
```

`data.names` lines up with `data.code`, and one of them must be the entry
point — `main.py` unless you set `parameters.entry`. The rest are importable
modules, so a real multi-file program works.

## Getting arguments in

Three tiers, smallest ceremony first.

**1 · Envelope-native, for bulk data.** Every client can already send these:

| you send | the script sees |
|---|---|
| `data.images` (`bb`) | `vs.images` — decoded BGR arrays; `vs.image` is the first one |
| | `vs.raw` — the original bytes, same order |
| `data.texts` (`ss`) | `vs.texts` — list of `str` |
| `data.numbers` (`ff`) | `vs.numbers` — list of `float` |

**2 · The request's `args` object, for named scalars.** The `Value` oneof has
six kinds and no room for `{"lo": 50, "hi": 150}`, so named arguments go in the
config section instead:

```python
config={"pycv": {"command": "run", "args": {"lo": 50, "hi": 150}}}
# in the script:  vs.args["lo"]
```

**3 · `data.inputs`, one bundle, for everything else.** A single `.npz` of
named arrays becomes `vs.inputs`, with dtypes and shapes intact and
`allow_pickle=False`:

```python
import numpy as np, io
buf = io.BytesIO(); np.savez(buf, K=K, pts=pts)
data={"inputs": buf.getvalue(), ...}        # vs.inputs["K"], vs.inputs["pts"]
```

A MATLAB `.mat` is read too **if** the image was built with scipy — the
published image is not, because scipy is ~138 MB against a 400 MB image and
`.npz` covers the same ground losslessly. If you need it, uncomment `scipy` in
`requirements.txt` and rebuild; a `.mat` arriving at an image without it gets a
reply that says exactly that. From MATLAB or Octave, the other direction is
easier than it sounds: a `.npz` is a zip of `.npy` files, so `unzip` plus
`readNPY` from [npy-matlab](https://github.com/kwikteam/npy-matlab) reads a
reply, and `writeNPY` plus `zip` builds a bundle.

## Getting results out

The script calls `vs.emit(name, value)` as often as it likes. The value's type
picks the codec the box declares for that field, so the client decodes it
without knowing anything about this box:

| the script emits | codec | the client gets |
|---|---|---|
| `np.ndarray` | `numpy` | the array, dtype and shape intact |
| `bytes` | `identity` | raw bytes — an encoded PNG, a file |
| `str`, number, `bool`, `dict`, `list`, `None` | `json` | the value |
| anything else | — | an error naming the field |

That last row is on purpose: the box never pickles on your behalf. On a box
that already runs arbitrary code, an implicit second execution channel is not
a convenience worth having.

`stdout` and `stderr` always come back, whatever the outcome — a script that
fails is far more useful with its traceback attached than with a bare status.

## Parameters

| key | default | meaning |
|---|---|---|
| `timeout` | 30 | wall-clock seconds; capped at 600. Past it the process group is killed and the reply carries `timed_out: true` |
| `entry` | `main.py` | which sent file is executed (`run` only) |
| `max_memory_mb` | 2048 | `RLIMIT_AS`, so a runaway allocation fails as a clean `MemoryError`. `0` disables it |
| `max_output_mb` | 256 | biggest file the script may write |
| `decode_images` | `true` | `false` leaves `vs.images` as the original bytes |

Also enforced: 1 MB of source across all files, 64 files, `.py` names with no
`..` or absolute paths, and 64 000 characters of each log stream.

## Reply

```json
{"pycv": {"status": "done", "command": "run", "entry": "main.py",
          "emitted": ["edges", "meta"], "returncode": 0,
          "runtime": 0.41, "timeout": 30,
          "encoding": {"edges": "numpy", "meta": "json",
                       "stdout": "identity", "stderr": "identity"}}}
```

`status` is `done`, `empty_request` (no `data.code`) or `error`. A script that
exits non-zero is an `error` whose message carries the last line of its
traceback, with the full `stderr` in the data.

## What is in the image

`numpy` and `opencv-python-headless`, pinned `>=4.10,<5`. Nothing else — no
torch, no CUDA, no scikit-anything. The pin matters: an unpinned install
resolves to OpenCV 5.0 today, and user code is overwhelmingly written against
the 4.x API, so a silent major-version jump would break scripts that have
nothing to do with this box.

`cv2.ml` (KMeans, SVM, KNN, RTrees, EM) and `cv2.dnn` (ONNX inference on CPU)
are both there, so a fair amount of classical work needs no extra package. The
`contrib` modules — `aruco`, `ximgproc`, `xfeatures2d` — are **not**: swapping
`opencv-python-headless` for `opencv-contrib-python-headless` costs about
97 MB if you want them.

## Example: a multi-file script

```python
main = b'''
import vs, helper
vs.emit("edges", helper.edges(vs.image, vs.args["lo"], vs.args["hi"]))
vs.emit("meta", {"shape": list(vs.image.shape), "n": len(vs.images)})
'''
helper = b'''
import cv2
def edges(img, lo, hi):
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), lo, hi)
'''
res = b.run(data={"code": [main, helper], "names": ["main.py", "helper.py"],
                  "images": [pathlib.Path("dog.jpg")]},
            config={"pycv": {"command": "run", "args": {"lo": 50, "hi": 150}}})
res.edges      # numpy array
res.meta       # dict
res.stdout     # whatever it printed
```

## Test

```bash
python3 test/test_pycv.py                    # starts its own server, no docker
python3 test/test_pycv.py --host localhost:9075   # against a running box
```

20 checks: both commands, all three argument tiers, every codec, and the
failure paths — missing entry point, crash, timeout, process-group kill,
memory cap, unsendable value.
