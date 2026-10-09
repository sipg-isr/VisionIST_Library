"""Shared helpers: find the repo, load box manifests, resolve image references.

Every other tool builds on this. Nothing here knows about any particular box -
the knowledge is in boxes/<name>/box.yaml.
"""

from __future__ import annotations

import json
import pathlib
import sys

try:
    import yaml
except ImportError:                                        # pragma: no cover
    sys.exit("PyYAML is required:  pip install -r tools/requirements.txt")

#: Registry host per short name. A box names a repository; the caller picks
#: where to pull it from, so the same fleet definition works off either.
REGISTRY_HOSTS = {
    "ghcr": "ghcr.io",
    "dockerhub": "docker.io",
}

#: Committed test fixtures above this are rejected: the repo is cloned by
#: everyone, forever, including the history. Bigger files belong in `assets:`.
MAX_FIXTURE_BYTES = 256 * 1024


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """The VisionIST_Library checkout containing this file."""
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "boxes").is_dir() and (parent / "contract").is_dir():
            return parent
    raise SystemExit("not inside a VisionIST_Library checkout "
                     "(no boxes/ + contract/ above this file)")


def box_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in (root / "boxes").iterdir()
                  if p.is_dir() and (p / "box.yaml").is_file())


def load_manifest(box_dir: pathlib.Path) -> dict:
    with (box_dir / "box.yaml").open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{box_dir.name}/box.yaml is not a mapping")
    data["_dir"] = box_dir.name
    return data


def load_all(root: pathlib.Path) -> list[dict]:
    return [load_manifest(d) for d in box_dirs(root)]


#: The only tag this registry publishes or pulls. A box's `version` in its
#: manifest is the version of the RECIPE - it drives the release git tag and
#: the "has this box changed?" checks - but it never lands in an image name.
#: One moving tag per box means a fleet compose file never goes stale, and
#: `docker compose pull` is the whole upgrade procedure.
IMAGE_TAG = "latest"


def image_ref(manifest: dict, registry: str = "dockerhub",
              tag: str | None = None) -> str:
    """Full pullable reference for a box on one registry.

    Always ``:latest`` unless a caller explicitly overrides ``tag``: image
    names in this project carry no version (see :data:`IMAGE_TAG`).

    ``ghcr`` keeps the repository path as given; ``dockerhub`` flattens it,
    because Docker Hub has exactly one level of namespace - sipgisr/visionist-clip
    becomes sipgisr/visionist-clip.
    """
    if registry not in REGISTRY_HOSTS:
        raise ValueError(f"unknown registry {registry!r} "
                         f"(known: {', '.join(sorted(REGISTRY_HOSTS))})")
    repo = manifest["image"]["repository"]
    tag = tag or IMAGE_TAG
    if registry == "dockerhub":
        org, _, name = repo.rpartition("/")
        repo = f"{org.replace('-', '').replace('/', '')}/{name}" if org else name
    return f"{REGISTRY_HOSTS[registry]}/{repo}:{tag}"


#: Where the host-port ledger starts. Only used when the ledger is empty.
BASE_PORT = 9061

#: The ledger itself: box name -> permanent host port.
PORTS_FILE = "registry/ports.json"


def load_ports(root: pathlib.Path) -> dict:
    """Read registry/ports.json. Missing file -> an empty ledger."""
    path = root / PORTS_FILE
    if not path.is_file():
        return {"generated_by": "tools/build_index.py", "base_port": BASE_PORT,
                "ports": {}}
    return json.loads(path.read_text())


def assign_ports(ledger: dict, manifests: list[dict]) -> tuple[dict, list[str]]:
    """Give every box a host port, assigning sequentially to new ones.

    Ports are issued in ARRIVAL order, not alphabetical order: a box keeps the
    port it was first given, forever, and a new box takes the next port after
    the highest ever issued. Alphabetical assignment moved every box after the
    newcomer, which silently repointed every client config and every
    hand-written host list in the fleet.

    A retired box's port is NOT recycled - reusing it would point an old client
    at a different box, which fails as a wrong answer rather than as a refused
    connection.

    Returns the updated ledger and the names of the boxes newly assigned.
    """
    ports = dict(ledger.get("ports") or {})
    base = int(ledger.get("base_port") or BASE_PORT)
    new = []
    for m in sorted(manifests, key=lambda x: x["name"]):
        if m["name"] in ports:
            continue
        ports[m["name"]] = max(ports.values(), default=base - 1) + 1
        new.append(m["name"])
    out = dict(ledger)
    out.setdefault("generated_by", "tools/build_index.py")
    out["base_port"] = base
    out["ports"] = ports
    return out, new


def registries_of(manifest: dict) -> list[str]:
    return manifest.get("image", {}).get("registries") or ["dockerhub"]
