"""Fetch an InternVL3 checkpoint at image build time.

A box must not reach the network at request time, so the checkpoint is baked
into the image here and the service runs with HF_HUB_OFFLINE=1.

Build args (passed through as environment):
  MODEL_ID          Hugging Face repo id         (OpenGVLab/InternVL3-1B)
  MODEL_REVISION    branch, tag or commit sha    (main)
  DOWNLOAD_WEIGHTS  false -> fetch nothing; mount a checkpoint at
                    /weights/model when you run the box

The repo id is written to /weights/model/MODEL_ID so replies can say which
model answered.
"""

import os
import sys

MODEL_ID = os.environ.get("MODEL_ID", "OpenGVLab/InternVL3-1B")
REVISION = os.environ.get("MODEL_REVISION", "main")
WANT = os.environ.get("DOWNLOAD_WEIGHTS", "true").strip().lower()
DEST = "/weights/model"

os.makedirs(DEST, exist_ok=True)

if WANT not in ("true", "1", "yes", "on"):
    print(f"DOWNLOAD_WEIGHTS={WANT!r}: not fetching. Mount an InternVL "
          f"checkpoint at {DEST} when you run the box.")
    sys.exit(0)

from huggingface_hub import snapshot_download  # noqa: E402

print(f"fetching {MODEL_ID}@{REVISION}", flush=True)
try:
    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=DEST,
        # The weights, configs, tokenizer and the remote modelling code -
        # not the README images or alternative weight formats.
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt",
                        "*.model", "*.tiktoken", "*.jinja"],
    )
except Exception as e:                                      # noqa: BLE001
    sys.exit(f"could not fetch {MODEL_ID}@{REVISION}: {e}\n"
             f"If this host cannot reach huggingface.co, build with "
             f"--build-arg DOWNLOAD_WEIGHTS=false and mount {DEST}.")

# snapshot_download leaves a .cache folder of download metadata behind.
cache = os.path.join(DEST, ".cache")
if os.path.isdir(cache):
    import shutil
    shutil.rmtree(cache)

with open(os.path.join(DEST, "MODEL_ID"), "w") as f:
    f.write(MODEL_ID + "\n")

total = sum(os.path.getsize(os.path.join(root, n))
            for root, _, names in os.walk(DEST) for n in names)
print(f"checkpoint ready in {DEST} ({total / 1e9:.2f} GB)")
