"""``vs`` - what a pycv script sees.

The box copies this file into each run directory as ``vs.py``, so a script
does ``import vs`` and finds its inputs already loaded and a way to hand
results back. Nothing here talks to gRPC: the box has written a few files
into the directory, and this module reads them.

    import vs, cv2
    g = cv2.cvtColor(vs.image, cv2.COLOR_BGR2GRAY)
    vs.emit("edges", cv2.Canny(g, vs.args.get("lo", 100), vs.args.get("hi", 200)))

What is available
-----------------
``vs.images``   list of decoded ``np.ndarray`` (BGR), from ``data.images``
``vs.image``    ``images[0]`` or ``None`` - the common case, spelled short
``vs.raw``      the undecoded bytes of each input image, same order
``vs.texts``    list of ``str``, from ``data.texts``
``vs.numbers``  list of ``float``, from ``data.numbers``
``vs.args``     dict, from the request's ``args`` section - your own arguments
``vs.params``   dict, the box's own knobs (timeout, entry, ...)
``vs.inputs``   dict of named arrays, from the ``data.inputs`` bundle
``vs.workdir``  ``pathlib.Path`` of this run's directory

``vs.emit(name, value)`` hands one result back. Call it as often as you like.
The value's Python type picks the codec the box declares to the client:

    np.ndarray                 -> "numpy"     (an np.save blob)
    bytes                      -> "identity"  (an encoded PNG/JPEG, or any file)
    str, int, float, bool,     -> "json"
    dict, list, tuple, None

Anything else raises, naming the field - the box never pickles on your behalf.
"""

from __future__ import annotations

import atexit
import json
import pathlib
import re

import numpy as np

__all__ = ["images", "image", "raw", "texts", "numbers", "args", "params",
           "inputs", "workdir", "emit", "emitted"]

workdir = pathlib.Path(__file__).resolve().parent
_OUT = workdir / "_vs_out"
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Names the box uses for its own reply fields; a script may not take them.
_RESERVED = {"stdout", "stderr"}

_env = json.loads((workdir / "_vs_env.json").read_text())

args: dict = _env.get("args") or {}
params: dict = _env.get("params") or {}
texts: list = list(_env.get("texts") or [])
numbers: list = [float(x) for x in (_env.get("numbers") or [])]

# ---- images ---------------------------------------------------------------
raw: list = []
images: list = []
for _rel in _env.get("image_files") or []:
    _blob = (workdir / _rel).read_bytes()
    raw.append(_blob)
    _img = None
    if _env.get("decode_images", True):
        try:
            import cv2
            _img = cv2.imdecode(np.frombuffer(_blob, dtype=np.uint8),
                                cv2.IMREAD_UNCHANGED)
        except Exception:                                       # noqa: BLE001
            _img = None
    images.append(_img if _img is not None else _blob)
image = images[0] if images else None

# ---- the bundle -----------------------------------------------------------
inputs: dict = {}
_bundle = _env.get("inputs_file")
if _bundle:
    _path = workdir / _bundle
    _head = _path.read_bytes()[:4]
    if _head[:2] == b"PK":                       # a .npz is a zip of .npy
        with np.load(_path, allow_pickle=False) as _z:
            inputs = {k: _z[k] for k in _z.files}
    else:                                        # a MATLAB .mat, if scipy is here
        try:
            import scipy.io
        except ImportError:                                     # pragma: no cover
            raise RuntimeError(
                "data.inputs looks like a MATLAB .mat, but this image was built "
                "without scipy. Send a .npz instead, or rebuild the box with "
                "scipy in requirements.txt (see the box README).")
        inputs = {k: v for k, v in scipy.io.loadmat(_path).items()
                  if not k.startswith("__")}

# ---- emitting -------------------------------------------------------------
emitted: dict = {}        # name -> codec, in call order
_json_values: dict = {}


def _flush_json():
    if _json_values:
        (_OUT / "_vs_json.json").write_text(json.dumps(_json_values))


atexit.register(_flush_json)


def emit(name, value):
    """Hand one result back to the caller. See the module docstring."""
    name = str(name)
    if not _NAME_RE.match(name):
        raise ValueError(
            f"emit(): {name!r} is not a usable field name - letters, digits "
            f"and underscores, not starting with a digit")
    if name in _RESERVED:
        raise ValueError(f"emit(): {name!r} is reserved by the box")
    _OUT.mkdir(exist_ok=True)

    if isinstance(value, np.generic):
        # np.float64 is a float subclass but np.int64 is not; .item() first so
        # both land in the json branch as plain Python scalars.
        value = value.item()

    if isinstance(value, np.ndarray):
        np.save(_OUT / f"{name}.npy", value, allow_pickle=False)
        emitted[name] = "numpy"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        (_OUT / f"{name}.bin").write_bytes(bytes(value))
        emitted[name] = "identity"
    elif isinstance(value, (str, int, float, bool, dict, list, tuple)) or value is None:
        if isinstance(value, tuple):
            value = list(value)
        try:
            json.dumps(value)
        except TypeError as e:
            raise TypeError(
                f"emit({name!r}): not JSON-serializable ({e}). Emit an "
                f"np.ndarray, bytes, or a plain JSON value.") from None
        _json_values[name] = value
        emitted[name] = "json"
    else:
        raise TypeError(
            f"emit({name!r}): cannot send a {type(value).__name__}. Emit an "
            f"np.ndarray, bytes, or a plain JSON value - this box never "
            f"pickles on your behalf.")
    (_OUT / "_vs_manifest.json").write_text(json.dumps(emitted))
