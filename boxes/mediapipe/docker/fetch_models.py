#!/usr/bin/env python3
"""Download the MediaPipe .task bundles this box serves.

Run by the image's `models` build stage, and usable by hand from a checkout:

    python3 docker/fetch_models.py models/          # the five bundles
    MODEL_VERSION=1 python3 docker/fetch_models.py models/

Why it exists: MediaPipe Tasks does not ship weights in its pip wheel, and a
box must not reach the network while serving a request. So the bundles are
fetched once at build time and copied into the image.

MODEL_VERSION is the upstream path segment. "latest" is what the MediaPipe
documentation publishes and is a MOVING alias; pass 1 for a build whose
weights are reproducible byte for byte.

`tools/fetch_assets.py mediapipe` does the same job from the `assets:` block
in box.yaml - this script is what the Dockerfile uses, so the image build does
not need the registry checked out.
"""

import hashlib
import os
import pathlib
import sys
import urllib.request

BASE = os.environ.get("MODEL_BASE", "https://storage.googleapis.com/mediapipe-models")
VERSION = os.environ.get("MODEL_VERSION", "latest")

#: bundle file name -> upstream directory, minus the version segment
BUNDLES = {
    "hand_landmarker.task": "hand_landmarker/hand_landmarker/float16",
    "pose_landmarker_lite.task": "pose_landmarker/pose_landmarker_lite/float16",
    "pose_landmarker_full.task": "pose_landmarker/pose_landmarker_full/float16",
    "pose_landmarker_heavy.task": "pose_landmarker/pose_landmarker_heavy/float16",
    "face_landmarker.task": "face_landmarker/face_landmarker/float16",
}


def main() -> int:
    dest = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    dest.mkdir(parents=True, exist_ok=True)

    for name, path in BUNDLES.items():
        url = f"{BASE}/{path}/{VERSION}/{name}"
        target = dest / name
        if target.is_file() and target.stat().st_size:
            print(f"  {name}: already here ({target.stat().st_size} bytes)")
            continue
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as e:
            print(f"  {name}: FAILED from {url} — {e}", file=sys.stderr)
            return 1
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"  {name}: {target.stat().st_size} bytes  sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
