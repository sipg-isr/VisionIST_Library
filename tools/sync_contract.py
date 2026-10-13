#!/usr/bin/env python3
"""Copy contract/ into every box's protos/, or check that it is already there.

Boxes keep their own protos/ so a box directory stays self-contained: copy it
anywhere and it builds. But the copy is GENERATED, never edited - this script
writes it and CI runs `--check` to fail any PR where one has drifted.

That matters more than it looks. In the 11-box VisionIST repo this registry was
split out of, pipeline.proto had already forked into three variants. They were
cosmetic (same field numbers, same wire format) and nobody noticed - which is
the point. The next fork would not have been cosmetic, and would have surfaced
as an unexplainable decode error in somebody else's fleet.

    python3 tools/sync_contract.py            # write
    python3 tools/sync_contract.py --check    # verify (CI)
"""

import argparse
import filecmp
import shutil
import sys

from _registry import box_dirs, repo_root

#: What every box gets a copy of. The generated *_pb2.py files are NOT here:
#: they are produced inside the Docker build from the .proto, so there is no
#: generated Python to keep in sync. serve.py holds the server limits every
#: box reads from the environment - shared for the same reason as aux.py:
#: sixteen hand-maintained copies is how they drift.
CONTRACT_FILES = ["pipeline.proto", "aux.py", "serve.py"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any copy differs")
    args = ap.parse_args()

    root = repo_root()
    contract = root / "contract"
    missing = [f for f in CONTRACT_FILES if not (contract / f).is_file()]
    if missing:
        print(f"contract/ is missing {', '.join(missing)}", file=sys.stderr)
        return 2

    drifted, written = [], []
    for box in box_dirs(root):
        protos = box / "protos"
        protos.mkdir(exist_ok=True)
        for fname in CONTRACT_FILES:
            src, dst = contract / fname, protos / fname
            same = dst.is_file() and filecmp.cmp(src, dst, shallow=False)
            if same:
                continue
            if args.check:
                drifted.append(f"{box.name}/protos/{fname}")
            else:
                shutil.copy2(src, dst)
                written.append(f"{box.name}/protos/{fname}")

    if args.check:
        if drifted:
            print("these copies differ from contract/ "
                  "(run tools/sync_contract.py):", file=sys.stderr)
            for d in drifted:
                print(f"  boxes/{d}", file=sys.stderr)
            return 1
        print(f"contract in sync across {len(box_dirs(root))} box(es)")
        return 0

    if written:
        print(f"updated {len(written)} file(s):")
        for w in written:
            print(f"  boxes/{w}")
    else:
        print(f"already in sync across {len(box_dirs(root))} box(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
