"""Fetch the ABot-Recon checkpoint (and optionally the loop-closure assets)
at image build time.

Kept as a file rather than inlined in the Dockerfile so the build fails with
a readable message rather than a shell quoting accident, and so the skip
paths (--build-arg DOWNLOAD_WEIGHTS=false / LOOP_ASSETS=false) are obvious.
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = os.environ.get("HF_REPO_ID", "acvlab/ABot-Recon")
WANT = os.environ.get("WANT_WEIGHTS", "true").strip().lower()
WANT_LOOP = os.environ.get("WANT_LOOP", "false").strip().lower()

WEIGHTS = Path("/weights")
LOOP_DIR = WEIGHTS / "loop"

# The upstream's own download script, same two files and URLs.
LOOP_ASSETS = (
    ("dino_salad.ckpt",
     "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"),
    ("dinov2_vitb14_pretrain.pth",
     "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"),
)


def fetch(url, dest: Path) -> None:
    print(f"fetching {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "VisionIST-Library/0.1"})
    try:
        with urllib.request.urlopen(req) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if tmp.stat().st_size == 0:
            raise RuntimeError("downloaded an empty file")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  -> {dest}  ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)


if WANT not in ("true", "1", "yes", "on"):
    print(f"DOWNLOAD_WEIGHTS={WANT!r}: not fetching. Mount the checkpoint at "
          f"{WEIGHTS}/abot_recon.safetensors when you run the box.")
else:
    from huggingface_hub import hf_hub_download  # noqa: E402

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name in ("abot_recon.safetensors", "config.json"):
        print(f"fetching {REPO}:{name}", flush=True)
        try:
            cached = hf_hub_download(repo_id=REPO, filename=name)
        except Exception as e:                                  # noqa: BLE001
            sys.exit(f"could not fetch {name} from {REPO}: {e}\n"
                     f"If this host cannot reach huggingface.co, build with "
                     f"--build-arg DOWNLOAD_WEIGHTS=false and mount "
                     f"{WEIGHTS}/abot_recon.safetensors.")
        shutil.copyfile(cached, WEIGHTS / name)
        print(f"  -> {WEIGHTS / name}  "
              f"({(WEIGHTS / name).stat().st_size / 1e9:.2f} GB)", flush=True)

if WANT_LOOP not in ("true", "1", "yes", "on"):
    print(f"LOOP_ASSETS={WANT_LOOP!r}: not fetching the loop-closure assets. "
          f"`loop_closure: true` needs them at {LOOP_DIR} (or a mount).")
else:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in LOOP_ASSETS:
        fetch(url, LOOP_DIR / name)

print("weights ready")
