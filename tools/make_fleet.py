#!/usr/bin/env python3
"""Pick boxes from the registry, get a runnable fleet.

Writes a docker-compose.yml that PULLS prebuilt images - nobody builds a
hundred boxes to run three of them - plus the fleet.json the webui reads.
Host ports are assigned automatically, so adding a box never means hunting
for a free port in a hand-maintained list.

    python3 tools/make_fleet.py --boxes clip,yolo,moge --out my-fleet
    python3 tools/make_fleet.py --tag features --registry dockerhub --out cpu-fleet
    python3 tools/make_fleet.py --all --runtime cpu --out laptop-fleet

Images are pulled as :latest - box image names carry no version, so a compose
file generated once never goes stale: `docker compose pull` is the upgrade.

The index can be local or remote, so this same script works from a
VisionIST_Library checkout or from a VisionIST one with no checkout at all:

    --index registry/index.json
    --index https://raw.githubusercontent.com/jpcosteira/VisionIST_Library/main/registry/index.json
"""

import argparse
import json
import pathlib
import sys
import urllib.request

DEFAULT_INDEX_URL = ("https://raw.githubusercontent.com/jpcosteira/"
                     "VisionIST_Library/main/registry/index.json")

#: Boxes that want a GPU reservation in compose.
GPU_RUNTIMES = {"gpu", "gpu-or-cpu", "cuda-only"}

GPU_BLOCK = """    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""


def load_index(where: str | None) -> dict:
    if where is None:
        local = pathlib.Path(__file__).resolve().parent.parent / "registry" / "index.json"
        where = str(local) if local.is_file() else DEFAULT_INDEX_URL
    if where.startswith(("http://", "https://")):
        with urllib.request.urlopen(where, timeout=30) as fh:
            return json.loads(fh.read().decode())
    path = pathlib.Path(where)
    if not path.is_file():
        sys.exit(f"no index at {path}")
    return json.loads(path.read_text())


def select(index: dict, args) -> list[dict]:
    boxes = index["boxes"]
    if args.boxes:
        wanted = [b.strip() for b in args.boxes.split(",") if b.strip()]
        by_name = {b["name"]: b for b in boxes}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            sys.exit(f"no such box: {', '.join(missing)}\n"
                     f"available: {', '.join(sorted(by_name))}")
        boxes = [by_name[w] for w in wanted]
    elif args.tag:
        boxes = [b for b in boxes if args.tag in b.get("tags", [])]
        if not boxes:
            sys.exit(f"no box carries the tag {args.tag!r}")
    elif not args.all:
        sys.exit("pick boxes with --boxes, --tag or --all")

    if args.runtime == "cpu":
        boxes = [b for b in boxes if b["runtime"] in ("cpu", "gpu-or-cpu")]
    return sorted(boxes, key=lambda b: b["name"])


def resolve_ref(box: dict, registry: str) -> tuple[str, str | None]:
    """The image to pull, plus a note when we had to fall back."""
    refs = box["image"]["refs"]
    if registry in refs:
        return refs[registry], None
    other = next(iter(refs))
    return refs[other], (f"{box['name']}: not published on {registry}, "
                         f"using {other}")


#: Compose settings per declared security profile (registry/schema.json).
#: Keyed by profile name, never by box name - a box asks for containment in
#: its manifest and the generator knows what that means.
SECURITY_PROFILES = {
    "standard": [],
    "untrusted-code": [
        "    # This box executes code callers send it (box.yaml security.profile).",
        "    # These bound an accident; they are not a sandbox, and Docker cannot",
        "    # give a container published ports AND no egress - block outbound",
        "    # traffic at the host if you need it.",
        "    read_only: true",
        "    tmpfs: [/tmp]",
        "    cap_drop: [ALL]",
        "    security_opt: [no-new-privileges:true]",
    ],
}


def security_lines(box: dict) -> list:
    """Compose hardening for one box, from its declared profile."""
    sec = box.get("security") or {}
    lines = list(SECURITY_PROFILES.get(sec.get("profile", "standard"), []))
    if not lines:
        return lines
    if sec.get("mem_limit"):
        lines.append(f"    mem_limit: {sec['mem_limit']}")
    if sec.get("pids_limit"):
        lines.append(f"    pids_limit: {sec['pids_limit']}")
    return lines


def port_of(box: dict, args, fallback: dict) -> int:
    """The box's host port.

    Ports are a property of the BOX, not of the fleet: they come from the
    registry's ledger (registry/ports.json, surfaced as `port` in the index)
    and are issued in arrival order, so a box answers on the same host port in
    every fleet anyone generates, full or partial. An older published index
    that predates the ledger has no `port` - only then does this fall back to
    counting from --base-port in name order.
    """
    port = box.get("port")
    return int(port) if port else fallback[box["name"]]


def compose(boxes: list[dict], args) -> tuple[str, list[str]]:
    lines = [
        "# Generated by tools/make_fleet.py - edit the command, not this file.",
        "#",
        f"#   boxes:    {', '.join(b['name'] for b in boxes)}",
        f"#   registry: {args.registry}",
        "#",
        "# Each service pulls a prebuilt image. Every box listens on 8061 inside",
        "# its container (AI4EU spec); the host port below is the box's own,",
        "# issued once in arrival order and recorded in the registry's",
        "# ports.json - so it is the same in every fleet, and a partial fleet",
        "# has gaps rather than renumbered boxes. On the fleet network a box is",
        "# reachable by its service name on 8061, which is what fleet.json uses.",
        "",
        "name: visionist-fleet",
        "",
        "services:",
    ]
    notes = []
    fallback = {b["name"]: args.base_port + i for i, b in enumerate(boxes)}
    if any(not b.get("port") for b in boxes):
        notes.append("this index predates registry/ports.json: ports counted "
                     "from --base-port in name order")
    for box in boxes:
        ref, note = resolve_ref(box, args.registry)
        if note:
            notes.append(note)
        size = box["image"].get("approx_size_gb")
        lines.append(f"  # {box['summary']}"
                     + (f"  (~{size} GB)" if size else ""))
        lines.append(f"  {box['name']}:")
        lines.append(f"    image: {ref}")
        lines.append(f"    container_name: visionist-{box['name']}")
        lines.append(f'    ports: ["{port_of(box, args, fallback)}:8061"]')
        lines.append("    environment: [ PORT=8061 ]")
        lines.append("    restart: unless-stopped")
        lines.extend(security_lines(box))
        if box["runtime"] in GPU_RUNTIMES and args.gpu:
            lines.append(GPU_BLOCK.rstrip("\n"))
        if box["runtime"] == "cuda-only" and not args.gpu:
            notes.append(f"{box['name']} is CUDA-only and will fail without --gpu")
        lines.append("")

    if args.webui_image:
        depends = "\n".join(f"      - {b['name']}" for b in boxes)
        lines += [
            "  # The browser front end. It reaches boxes by SERVICE NAME on 8061,",
            "  # which is what data/fleet.json below records.",
            "  webui:",
            f"    image: {args.webui_image}",
            "    container_name: visionist-webui",
            f'    ports: ["{args.webui_port}:8000"]',
            "    environment:",
            "      - WEBUI_HOST=0.0.0.0",
            "      - WEBUI_PORT=8000",
            "      - WEBUI_DATA_DIR=/data",
            "    extra_hosts:",
            "      - host.docker.internal:host-gateway",
            "    volumes:",
            "      - ./data:/data",
            "    depends_on:",
            depends,
            "    restart: unless-stopped",
            "",
        ]
    return "\n".join(lines), notes


def fleet_json(boxes: list[dict]) -> dict:
    return {"entries": [{"id": b["name"], "name": b["name"],
                         "addr": f"{b['name']}:8061", "def_id": b["key"],
                         "note": b["summary"], "last_probe": None}
                        for b in boxes]}


def readme(boxes: list[dict], args, ports: dict) -> str:
    rows = "\n".join(
        f"| `{b['name']}` | `{b['key']}` | {b['runtime']} | {ports[b['name']]} | "
        f"{b['summary']} |" for b in boxes)
    gpu = "with" if args.gpu else "without"
    return f"""# VisionIST fleet

Generated by `tools/make_fleet.py` from the
[VisionIST_Library](https://github.com/jpcosteira/VisionIST_Library) registry,
pulling from **{args.registry}**, {gpu} GPU reservations.

```bash
docker compose pull      # no building: these are published images
docker compose up -d
docker compose ps
```

| Box | Config key | Runtime | Host port | What it does |
|---|---|---|---|---|
{rows}

Call one with the client:

```python
from visionist_client import Visionist
b = Visionist("localhost:{min(ports.values())}")
```

To change the selection, re-run the generator rather than editing
`docker-compose.yml` - it is overwritten.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    pick = ap.add_argument_group("choosing boxes")
    pick.add_argument("--boxes", help="comma-separated names, in order")
    pick.add_argument("--tag", help="every box carrying this tag")
    pick.add_argument("--all", action="store_true")
    pick.add_argument("--runtime", choices=["any", "cpu"], default="any",
                      help="cpu drops boxes that need a GPU")

    ap.add_argument("--out", required=True, help="directory to write")
    ap.add_argument("--index", default=None,
                    help="registry/index.json path or URL "
                         "(default: local if present, else the published one)")
    ap.add_argument("--registry", choices=["ghcr", "dockerhub"], default="dockerhub")
    ap.add_argument("--base-port", type=int, default=9061,
                    help="only used with an index that predates "
                         "registry/ports.json; ports otherwise come from the box")
    ap.add_argument("--gpu", action="store_true",
                    help="add nvidia reservations for boxes that want one")
    ap.add_argument("--webui-image", default=None,
                    help="also run the webui from this image")
    ap.add_argument("--webui-port", type=int, default=8080)
    args = ap.parse_args()

    index = load_index(args.index)
    boxes = select(index, args)

    out = pathlib.Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    text, notes = compose(boxes, args)
    (out / "docker-compose.yml").write_text(text)
    (out / "data" / "fleet.json").write_text(json.dumps(fleet_json(boxes), indent=2) + "\n")
    fallback = {b["name"]: args.base_port + i for i, b in enumerate(boxes)}
    ports = {b["name"]: port_of(b, args, fallback) for b in boxes}
    (out / "README.md").write_text(readme(boxes, args, ports))

    print(f"{len(boxes)} box(es) -> {out}/")
    for b in boxes:
        print(f"  {b['name']:18s} {ports[b['name']]}  {resolve_ref(b, args.registry)[0]}")
    for n in notes:
        print(f"  note: {n}", file=sys.stderr)
    print(f"\n  cd {out} && docker compose pull && docker compose up -d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
