"""pycv box - runs user-supplied Python against OpenCV.

    {"pycv": {"command": "run" | "eval" | "reset",
              "parameters": {"timeout": 30, "entry": "main.py", ...},
              "args": { ... your script's own arguments ... }}}

``run``   ``data.code`` is a BytesList of .py files and ``data.names`` the
          matching StringList of file names; exactly one must be the entry
          point (``main.py`` by default).
``eval``  ``data.code`` is one string: an expression such as
          ``cv2.GaussianBlur(vs.image, (5, 5), 1.5)``, or a few statements.
          An expression's value is emitted as ``result``; a tuple is unpacked
          into ``result_0``, ``result_1``, ... because so much of the cv2 API
          returns one.
``reset`` accepted and a no-op: this box keeps no state between calls.

SECURITY. This box executes code it is sent. Anyone who can reach its port can
run anything its user can run. That is the whole point of the box, not a
defect, but it makes it categorically different from every other box in the
registry: it belongs on a trusted network, never on a public one. What this
file does do is bound the damage of an *accident* - a runaway loop, a
fork bomb, a 40 GB allocation, a segfault - by running each request in a
subprocess with a timeout and rlimits, so one bad script cannot take the box
down for everyone else. See the README for the container-level settings.

How arguments arrive, in three tiers:

* envelope-native - ``data.images`` / ``texts`` / ``numbers`` become
  ``vs.images`` / ``vs.texts`` / ``vs.numbers``. Every client can send these.
* the request's ``args`` object - free-form JSON, becomes ``vs.args``. This is
  where named scalars belong; the Value oneof has no place for them.
* ``data.inputs`` - one ``.npz`` (or ``.mat``, if the image was built with
  scipy) holding named arrays, becomes ``vs.inputs``.

Results come back by ``vs.emit(name, value)``, and the value's type picks the
codec this box declares for that field: ndarray -> numpy, bytes -> identity,
JSON value -> json. ``stdout`` and ``stderr`` always come back too, whatever
the outcome - a failing script is far more useful with its traceback attached.
"""

import concurrent.futures as futures
import json
import logging
import os
import pathlib
import re

import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.append("./protos")
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402

import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_PORT_ENV_VAR = "PORT"
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

#: The config section this box answers to (== `key` in box.yaml).
BOX_KEY = "pycv"

#: Where this file lives inside the image; vs_prelude.py sits beside it.
_HERE = pathlib.Path(__file__).resolve().parent

_DEFAULTS = {
    "timeout": 30,          # seconds of wall clock for the script
    "entry": "main.py",     # which of the sent files is run
    "max_memory_mb": 2048,  # address-space cap for the script
    "max_output_mb": 256,   # biggest file the script may write
    "decode_images": True,  # hand vs.images as arrays rather than bytes
}
_LIMITS = {
    "timeout": 600,         # a caller may not ask for more than this
    "max_memory_mb": 16384,
    "code_bytes": 1 << 20,  # 1 MB of source, all files together
    "num_files": 64,
    "log_chars": 64_000,    # stdout/stderr returned, per stream
}

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*\.py$")


# ---------------------------------------------------------------- envelopes
def _reply(section, data=None, encoding=None):
    if encoding:
        section = {**section, "encoding": encoding}
    return pipeline_pb2.Envelope(
        config_json=json.dumps({BOX_KEY: section}), data=data or {})


def _done(extra, data=None, encoding=None):
    return _reply({"status": "done", **extra}, data, encoding)


def _empty_request():
    return _reply({"status": "empty_request"})


def _error(message, extra=None):
    return _reply({"status": "error", "error": message, **(extra or {})})


def _truncate(text, limit=_LIMITS["log_chars"]):
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters]"


# ----------------------------------------------------------------- the work
def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray, str, float, int)):
        return [value]
    return list(value)


def _check_number(section, key, default, cap=None):
    value = section.get(key, default)
    try:
        value = type(default)(value)
    except (TypeError, ValueError):
        raise ValueError(f"parameters.{key} must be a number, got {value!r}")
    if value <= 0:
        raise ValueError(f"parameters.{key} must be positive, got {value}")
    if cap is not None and value > cap:
        raise ValueError(f"parameters.{key} is capped at {cap}, got {value}")
    return value


def _write_sources(rundir, command, code, names, entry):
    """Write the script files into the run directory; return the entry name."""
    if command == "eval":
        if len(code) != 1:
            raise ValueError(
                f"eval takes exactly one code string, got {len(code)}")
        expr = code[0]
        if isinstance(expr, (bytes, bytearray)):
            expr = bytes(expr).decode("utf-8")
        if not expr.strip():
            raise ValueError("eval: the code string is empty")
        (rundir / "main.py").write_text(_EVAL_WRAPPER.format(src=repr(expr)))
        return "main.py"

    if not names:
        raise ValueError(
            "run needs data.names: one file name per entry in data.code")
    if len(names) != len(code):
        raise ValueError(
            f"data.names has {len(names)} name(s) for {len(code)} file(s)")

    total = 0
    for name, blob in zip(names, code):
        name = str(name)
        if not _SAFE_NAME.match(name) or ".." in name or name.startswith("/"):
            raise ValueError(
                f"{name!r} is not a usable file name - a relative path ending "
                f"in .py, no '..'")
        blob = bytes(blob) if isinstance(blob, (bytes, bytearray)) else \
            str(blob).encode("utf-8")
        total += len(blob)
        if total > _LIMITS["code_bytes"]:
            raise ValueError(
                f"the code is over the {_LIMITS['code_bytes'] // 1024} KB limit")
        path = rundir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)

    if entry not in [str(n) for n in names]:
        raise ValueError(
            f"no {entry!r} among the files sent ({', '.join(str(n) for n in names)}). "
            f"Name one of them {entry!r}, or set parameters.entry.")
    return entry


_EVAL_WRAPPER = '''"""Generated by the pycv box from an `eval` request."""
import vs
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

_src = {src}
try:
    _code = compile(_src, "<eval>", "eval")
    _is_expression = True
except SyntaxError:
    _code = compile(_src, "<eval>", "exec")
    _is_expression = False

_ns = {{"vs": vs, "np": np, "cv2": cv2, "images": vs.images,
        "image": vs.image, "texts": vs.texts, "numbers": vs.numbers,
        "args": vs.args, "inputs": vs.inputs, "emit": vs.emit}}

_result = eval(_code, _ns) if _is_expression else exec(_code, _ns)

if _is_expression and _result is not None:
    if isinstance(_result, tuple):
        # cv2 returns tuples constantly (threshold, findContours, ...).
        for _i, _part in enumerate(_result):
            vs.emit(f"result_{{_i}}", _part)
    else:
        vs.emit("result", _result)
'''


def _write_env(rundir, request, section, params):
    """Write the inputs the prelude will load, and return the env dict."""
    data = request.data
    env = {
        "args": section.get("args") or {},
        "params": params,
        "texts": [str(t) for t in _as_list(
            unwrap_value(data["texts"]) if "texts" in data else None)],
        "numbers": [float(n) for n in _as_list(
            unwrap_value(data["numbers"]) if "numbers" in data else None)],
        "decode_images": bool(params["decode_images"]),
        "image_files": [],
        "inputs_file": None,
    }
    if not isinstance(env["args"], dict):
        raise ValueError("the request's 'args' must be an object")

    images = _as_list(unwrap_value(data["images"])) if "images" in data else []
    for i, blob in enumerate(images):
        rel = f"_vs_img_{i:03d}.bin"
        (rundir / rel).write_bytes(bytes(blob))
        env["image_files"].append(rel)

    if "inputs" in data:
        bundle = unwrap_value(data["inputs"])
        if isinstance(bundle, list):
            bundle = bundle[0] if bundle else None
        if bundle:
            (rundir / "_vs_inputs.bin").write_bytes(bytes(bundle))
            env["inputs_file"] = "_vs_inputs.bin"

    (rundir / "_vs_env.json").write_text(json.dumps(env))
    return env


#: Runs in the child, AFTER exec, so the limits are set by a fresh
#: single-threaded interpreter rather than by a preexec_fn hook. That
#: distinction is not cosmetic: preexec_fn runs between fork and exec, and
#: forking a process whose other threads hold locks - which is exactly what a
#: gRPC server is - gives a child that segfaults instead of running the
#: script. Setting the limits here costs nothing and is always safe.
_LAUNCHER = """import resource, runpy, sys
_mem, _cpu, _out, _entry = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if _mem > 0:
    resource.setrlimit(resource.RLIMIT_AS, (_mem * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu))
resource.setrlimit(resource.RLIMIT_FSIZE, (_out * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.argv = [_entry]
runpy.run_path(_entry, run_name="__main__")
"""


def _collect(rundir):
    """Read back whatever the script emitted."""
    out = rundir / "_vs_out"
    fields, encoding = {}, {}
    if not out.is_dir():
        return fields, encoding

    manifest_path = out / "_vs_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    json_path = out / "_vs_json.json"
    json_values = json.loads(json_path.read_text()) if json_path.is_file() else {}

    for name, codec in manifest.items():
        if codec == "numpy":
            path = out / f"{name}.npy"
            if path.is_file():
                fields[name] = wrap_value(path.read_bytes())
                encoding[name] = "numpy"
        elif codec == "identity":
            path = out / f"{name}.bin"
            if path.is_file():
                fields[name] = wrap_value(path.read_bytes())
                encoding[name] = "identity"
        elif codec == "json" and name in json_values:
            fields[name] = wrap_value(
                json.dumps(json_values[name]).encode("utf-8"))
            encoding[name] = "json"
    return fields, encoding


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def Process(self, request, context):
        start = time.time()
        rundir = None
        try:
            if not request.config_json:
                return _error("No config JSON")
            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get(BOX_KEY), dict):
                return _error(f"config section {BOX_KEY!r} missing or not an object")

            section = config[BOX_KEY]
            command = section.get("command") or "run"
            raw_params = section.get("parameters") or {}
            if not isinstance(raw_params, dict):
                return _error("parameters must be an object")

            if command == "reset":
                return _done({"action": "reset",
                              "note": "pycv keeps no state between calls"})
            if command not in ("run", "eval"):
                return _error(f"unknown command {command!r} "
                              f"(expected 'run', 'eval' or 'reset')")

            params = {
                "timeout": _check_number(raw_params, "timeout",
                                         _DEFAULTS["timeout"], _LIMITS["timeout"]),
                "max_memory_mb": _check_number(raw_params, "max_memory_mb",
                                               _DEFAULTS["max_memory_mb"],
                                               _LIMITS["max_memory_mb"]),
                "max_output_mb": _check_number(raw_params, "max_output_mb",
                                               _DEFAULTS["max_output_mb"]),
                "entry": str(raw_params.get("entry", _DEFAULTS["entry"])),
                "decode_images": bool(raw_params.get("decode_images",
                                                     _DEFAULTS["decode_images"])),
            }

            code = _as_list(unwrap_value(request.data["code"])) \
                if "code" in request.data else []
            if not code:
                return _empty_request()
            if len(code) > _LIMITS["num_files"]:
                return _error(f"{len(code)} files sent; the limit is "
                              f"{_LIMITS['num_files']}")
            names = _as_list(unwrap_value(request.data["names"])) \
                if "names" in request.data else []

            rundir = pathlib.Path(tempfile.mkdtemp(prefix="pycv_"))
            (rundir / "_vs_out").mkdir()
            shutil.copyfile(_HERE / "vs_prelude.py", rundir / "vs.py")

            entry = _write_sources(rundir, command, code, names, params["entry"])
            _write_env(rundir, request, section, params)

            env = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HOME": str(rundir),
                "TMPDIR": str(rundir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "OPENCV_IO_ENABLE_OPENEXR": "0",
                "MPLBACKEND": "Agg",
            }

            (rundir / "_vs_launch.py").write_text(_LAUNCHER)
            argv = [sys.executable, "_vs_launch.py",
                    str(int(params["max_memory_mb"])),
                    str(int(params["timeout"]) + 1),
                    str(int(params["max_output_mb"])), entry]

            # start_new_session puts the script in its own process group, so a
            # timeout can kill everything it spawned rather than just the one
            # process we can see.
            proc = subprocess.Popen(
                argv, cwd=str(rundir), env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True)
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=params["timeout"])
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = None
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                stdout, stderr = proc.communicate()

            fields, encoding = _collect(rundir)
            fields["stdout"] = wrap_value(_truncate(stdout))
            fields["stderr"] = wrap_value(_truncate(stderr))

            info = {"command": command, "entry": entry,
                    "emitted": sorted(encoding),
                    "returncode": returncode,
                    "runtime": round(time.time() - start, 3),
                    "timeout": params["timeout"]}

            if timed_out:
                return _reply({"status": "error",
                               "error": f"the script did not finish within "
                                        f"{params['timeout']} s",
                               "timed_out": True, **info}, fields, encoding)
            if returncode != 0:
                first = (stderr or "").strip().splitlines()
                return _reply({"status": "error",
                               "error": f"the script exited with code {returncode}"
                                        + (f": {first[-1]}" if first else ""),
                               **info}, fields, encoding)
            return _done(info, fields, encoding)

        except ValueError as e:
            return _error(str(e))
        except Exception as e:                                  # noqa: BLE001
            logging.exception("Error in Process")
            return _error(f"{type(e).__name__}: {e}")
        finally:
            if rundir is not None:
                shutil.rmtree(rundir, ignore_errors=True)


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
    logging.warning("pycv executes code it is sent - keep this port on a "
                    "trusted network")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[("grpc.max_send_message_length", -1),
                 ("grpc.max_receive_message_length", -1)])
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    grpc_reflection.enable_server_reflection(
        (pipeline_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)

    run_server(server)
