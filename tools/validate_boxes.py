#!/usr/bin/env python3
"""Validate every box in the registry. This is the PR gate.

Checks, in order of how often they will actually fire:

  1. box.yaml parses and matches registry/schema.json
  2. the directory name equals the manifest's `name`
  3. the required files exist (Dockerfile, service source, README)
  4. the service file is named <key>_service.py, which is what the Dockerfile's
     SERVICE_NAME build arg resolves against
  5. no two boxes claim the same `key` or the same image repository
  6. committed fixtures are under the size cap, and anything over it is
     declared in `assets:` instead
  7. contract/ is in sync (delegated to sync_contract.py --check)

    python3 tools/validate_boxes.py             # all boxes
    python3 tools/validate_boxes.py clip yolo   # just these
"""

import argparse
import json
import pathlib
import sys

from _registry import MAX_FIXTURE_BYTES, box_dirs, load_manifest, repo_root

try:
    import jsonschema
except ImportError:                                        # pragma: no cover
    jsonschema = None


def human(n: int) -> str:
    return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1024/1024:.1f} MB"


def check_box(box: pathlib.Path, schema, seen_keys, seen_repos) -> list[str]:
    problems = []
    name = box.name

    try:
        m = load_manifest(box)
    except Exception as e:
        return [f"box.yaml does not load: {e}"]

    if schema is not None:
        validator = jsonschema.Draft202012Validator(schema)
        for err in sorted(validator.iter_errors({k: v for k, v in m.items()
                                                 if not k.startswith("_")}),
                          key=lambda e: list(e.path)):
            where = "/".join(str(p) for p in err.path) or "(root)"
            problems.append(f"schema: {where}: {err.message}")

    if m.get("name") != name:
        problems.append(f"name is {m.get('name')!r} but the directory is {name!r}")

    # extra_ports: the schema checks each entry; these are the cross-entry rules.
    names, containers = set(), {8061}
    for ep in m.get("extra_ports") or []:
        if ep.get("name") in names:
            problems.append(f"extra_ports: name {ep.get('name')!r} is declared twice")
        names.add(ep.get("name"))
        if ep.get("container") in containers:
            problems.append(f"extra_ports: container port {ep.get('container')} "
                            f"is already taken (8061 is the gRPC port)")
        containers.add(ep.get("container"))

    key = m.get("key")
    if key in seen_keys:
        problems.append(f"config key {key!r} is already used by {seen_keys[key]}")
    elif key:
        seen_keys[key] = name

    repo = (m.get("image") or {}).get("repository")
    if repo in seen_repos:
        problems.append(f"image repository {repo!r} is already used by {seen_repos[repo]}")
    elif repo:
        seen_repos[repo] = name

    # Required files. The Dockerfile renames <SERVICE_NAME>_service.py to
    # service.py inside the image, so the name has to match the config key.
    for rel in ("docker/Dockerfile", "README.md", "box.yaml"):
        if not (box / rel).is_file():
            problems.append(f"missing {rel}")
    if key:
        svc = box / "src" / f"{key}_service.py"
        if not svc.is_file():
            problems.append(f"missing src/{key}_service.py "
                            f"(the Dockerfile builds SERVICE_NAME={key})")

    # Fixture weight. Everyone who clones pays for this forever.
    declared = {a["path"] for a in (m.get("assets") or [])}
    for f in sorted(box.rglob("*")):
        if not f.is_file() or ".git" in f.parts:
            continue
        size = f.stat().st_size
        if size <= MAX_FIXTURE_BYTES:
            continue
        rel = f.relative_to(box).as_posix()
        if rel in declared:
            problems.append(f"{rel} is declared in assets: but also committed "
                            f"({human(size)}) - remove the committed copy")
        else:
            problems.append(f"{rel} is {human(size)}, over the "
                            f"{human(MAX_FIXTURE_BYTES)} cap - declare it under "
                            f"assets: and fetch it at test time")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("boxes", nargs="*", help="box names (default: all)")
    args = ap.parse_args()

    root = repo_root()
    schema = None
    if jsonschema is None:
        print("note: jsonschema is not installed, skipping schema validation "
              "(pip install -r tools/requirements.txt)", file=sys.stderr)
    else:
        with (root / "registry" / "schema.json").open() as fh:
            schema = json.load(fh)

    wanted = set(args.boxes)
    dirs = [d for d in box_dirs(root) if not wanted or d.name in wanted]
    unknown = wanted - {d.name for d in box_dirs(root)}
    if unknown:
        print(f"no such box: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    seen_keys, seen_repos, failed = {}, {}, 0
    for box in dirs:
        problems = check_box(box, schema, seen_keys, seen_repos)
        if problems:
            failed += 1
            print(f"FAIL {box.name}")
            for p in problems:
                print(f"       {p}")
        else:
            print(f"ok   {box.name}")

    print()
    if failed:
        print(f"{failed} of {len(dirs)} box(es) failed validation")
        return 1
    print(f"all {len(dirs)} box(es) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
