"""Convert Python objects to and from the pipeline.Value oneof.

THIS FILE IS THE SOURCE OF TRUTH. Each box carries a copy at
boxes/<name>/protos/aux.py written by tools/sync_contract.py; CI fails any PR
where a copy has drifted. Do not edit a box's copy.

This is the permissive variant: it accepts ints (as floats, since the oneof has
no integer kind), empty lists, mixed int/float lists, and bytearrays. The
VisionIST repo this registry was split out of had two versions of this file in
circulation - one of them rejected ints, so whether `data={"n": 3}` worked
depended on which box you were calling. Accepting more can only turn a
TypeError into a successful call, so the superset is the safe canonical choice.
"""

import pipeline_pb2


def wrap_value(obj):
    """Wrap a Python object into a pipeline.Value."""
    # bool is a subclass of int, so it lands in the int branch and becomes
    # 1.0 / 0.0 rather than raising.
    if isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)
    elif isinstance(obj, int):
        return pipeline_pb2.Value(f=float(obj))

    elif isinstance(obj, list):
        if len(obj) == 0:
            # An empty list has no element type to infer from; BytesList is
            # the conventional empty.
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=[]))
        elif all(isinstance(v, (float, int)) for v in obj):
            return pipeline_pb2.Value(
                ff=pipeline_pb2.FloatList(values=[float(v) for v in obj]))
        elif all(isinstance(v, str) for v in obj):
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=obj))
        elif all(isinstance(v, (bytes, bytearray)) for v in obj):
            return pipeline_pb2.Value(
                bb=pipeline_pb2.BytesList(values=list(obj)))

    raise TypeError(f"Cannot wrap object of type {type(obj)}: {obj}")


def unwrap_value(val):
    """Unwrap a pipeline.Value into a plain Python object."""
    if val is None:
        return None

    kind = val.WhichOneof("kind")
    if kind == "f":
        return val.f
    if kind == "s":
        return val.s
    if kind == "b":
        return val.b
    if kind == "ff":
        return list(val.ff.values)
    if kind == "ss":
        return list(val.ss.values)
    if kind == "bb":
        return list(val.bb.values)
    return None
