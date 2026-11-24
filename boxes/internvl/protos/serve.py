"""Server limits, read from the environment at startup.

THIS FILE IS THE SOURCE OF TRUTH. Each box carries a copy at
boxes/<name>/protos/serve.py written by tools/sync_contract.py; CI fails any
PR where a copy has drifted. Do not edit a box's copy.

Every box used to build its gRPC server with the limits written into the
source: unlimited message size, and a thread pool with no `max_workers`, which
Python quietly defaults to `min(32, cpu_count + 4)` - six on a two-core
machine. Neither could be changed without rebuilding the image, and the first
one is not a limit at all: one caller can send an arbitrarily large Envelope
and the box will try to hold all of it in memory.

So a box now asks this module instead, and both numbers come from the
environment - the same way `PORT` always has:

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=serve.max_workers()),
        options=serve.server_options())

Environment
-----------
``MAX_MESSAGE_MB``        largest Envelope accepted or sent, in MB.
                          Default 64. ``0`` means unlimited (the old
                          behaviour); do that only on a trusted network.
``MAX_WORKERS``           how many requests the box handles at once.
                          Default: the box's own, or Python's.
``MAX_CONCURRENT_RPCS``   queue depth before the server starts refusing.
                          Default: four times the worker count.

A caller that exceeds the message cap gets gRPC's own RESOURCE_EXHAUSTED,
which says what the limit was - it is a refusal, not a hang or a crash.
"""

import logging
import os

#: Big enough for a short video or a multi-view set, small enough that a
#: handful of concurrent requests cannot exhaust a shared server.
DEFAULT_MAX_MESSAGE_MB = 64


def _int_env(name, default):
    """Read a non-negative int from the environment, or keep the default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning("%s=%r is not an integer; using %r", name, raw, default)
        return default
    if value < 0:
        logging.warning("%s=%d is negative; using %r", name, value, default)
        return default
    return value


def max_message_bytes():
    """Message cap in bytes, or -1 for unlimited (``MAX_MESSAGE_MB=0``)."""
    mb = _int_env("MAX_MESSAGE_MB", DEFAULT_MAX_MESSAGE_MB)
    return -1 if mb == 0 else mb * 1024 * 1024


def max_workers(default=None):
    """Concurrent requests. ``default`` is the box's own preference; None
    leaves Python's ``min(32, cpu_count + 4)``."""
    value = _int_env("MAX_WORKERS", default)
    return value or None


def max_concurrent_rpcs(workers=None):
    """How many requests may be in flight before the server refuses.

    Without a ceiling the queue grows without bound and a burst turns into
    memory pressure rather than a clear refusal. Four times the worker count
    leaves room for a queue while keeping the bound finite.
    """
    explicit = _int_env("MAX_CONCURRENT_RPCS", 0)
    if explicit:
        return explicit
    if workers is None:
        workers = max_workers()
    if workers is None:
        workers = min(32, (os.cpu_count() or 1) + 4)
    return workers * 4


def server_options():
    """The ``options=`` list for ``grpc.server``."""
    cap = max_message_bytes()
    return [
        ("grpc.max_send_message_length", cap),
        ("grpc.max_receive_message_length", cap),
        # One HTTP/2 connection can carry many concurrent streams; without a
        # bound, a single client can open as many as it likes.
        ("grpc.max_concurrent_streams", _int_env("MAX_CONCURRENT_STREAMS", 64)),
    ]


def describe(workers=None):
    """One line for the startup log, so the limits in force are visible."""
    cap = max_message_bytes()
    size = "unlimited" if cap < 0 else f"{cap // (1024 * 1024)} MB"
    w = max_workers(workers)
    return (f"limits: max message {size}, "
            f"workers {w if w is not None else 'auto'}, "
            f"max in-flight {max_concurrent_rpcs(w)}")
