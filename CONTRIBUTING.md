# Contributing a box

A box is one container, one gRPC service, one directory here. You own yours:
CODEOWNERS is generated from your manifest, so you review changes to it.

## Start

```bash
git clone https://github.com/jpcosteira/VisionIST_Library
cd VisionIST_Library
pip install -r tools/requirements.txt

python3 tools/new_box.py my_box \
    --summary "What it does, in one line" \
    --runtime gpu --tags depth,monocular --github your-handle
```

What you get already builds and answers correctly — it is a working box that
reports image sizes. Replace the one marked section in
`src/my_box_service.py` with your own work, and the first thing you debug is
your model rather than the envelope.

## The five rules

**1. The config section is yours; the envelope is not.** Your service reads
`{"my_box": {"command": ..., "parameters": {...}}}` and answers with the same
section carrying a `status` of `done` | `empty_request` | `error`. Never a
bare top-level status. Every box accepts `"command": "reset"`, even when there
is nothing to reset.

**2. Never edit `protos/`.** It is written by `tools/sync_contract.py` from
`contract/`, and CI fails any PR where a copy has drifted. Changing the
contract itself is a wire change for every box at once and needs its own
discussion.

**3. Declare how your payloads are encoded.** Put `"encoding"` in your reply
config — a codec name, or a `{field: codec}` map — from `identity`, `json`,
`numpy`, `torch`, `zstd_pickle`. This is what lets a client decode your output
without knowing anything about your box.

**4. Fixtures stay under 256 KB.** Anything larger goes in the manifest's
`assets:` with a URL and a sha256, and `tools/fetch_assets.py` pulls it at test
time. A git repository keeps every byte of every version forever, for everyone
who clones.

**5. The manifest is the single source.** The registry index, the README
table, `.gitignore`, `CODEOWNERS` and the CI build matrix are all generated
from `box.yaml`. Do not maintain a list of boxes anywhere else.

## Before you open a PR

```bash
python3 tools/validate_boxes.py my_box     # schema, naming, files, fixture sizes
python3 tools/sync_contract.py             # refresh protos/
python3 tools/build_index.py               # regenerate index, ports, README, CODEOWNERS
python3 tools/check_docs.py my_box         # your README builds and runs the way CI does
```

`build_index.py` also issues your box its **host port** and appends it to
`registry/ports.json`. Ports go out in arrival order, so yours is simply the
next one after the highest ever issued — never an alphabetical slot, which
would have shifted every box after you and silently repointed everyone's
client config. Commit `registry/ports.json` with the rest; the port is
permanent from then on, and a retired box's port is never reused.

Then build it and run the shared contract test:

```bash
cd boxes/my_box
docker build --tag my_box -f docker/Dockerfile --build-arg SERVICE_NAME=my_box .
docker run --rm -d -p 8061:8061 -e PORT=8061 --name my_box my_box
python3 ../../contract/conformance/test_conformance.py --box my_box --host localhost:8061
python3 test/test_my_box.py
```

CI runs the same things, and builds only the boxes your PR touched.

## Filling in box.yaml

Most of it is self-explanatory from `template_box/box.yaml`. Three fields
cause nearly all the confusion:

**`key` vs `name`.** `name` is the directory; `key` is the config section your
service dispatches on. Keep them equal. They were not always: this registry's
`sbert` box lived in a directory called `textEmbedding`, and every client
needed a comment explaining it.

**`session.location`.** If your box is stateful, say where it reads
`session_id` from — the top of the config section, or inside `parameters`.
Both exist in this registry, and a client that guesses wrong silently gets the
default session instead of an error.

**`data.out[].codec`.** Must match what your service actually declares at
runtime. The conformance test checks the codec names are known and that the
declared fields exist; it cannot check that `numpy` really is an `np.save`
blob, so get it right.

## Releasing

Bump `version` in your manifest, then tag:

```bash
git tag my_box-v0.2.0 && git push origin my_box-v0.2.0
```

CI refuses the tag if it disagrees with the manifest, then builds and pushes
to Docker Hub (`sipgisr/`). Boxes version independently — your release does not
move anyone else's.

Running your own fork of this registry (other GitHub owner, other Docker Hub
namespace)? [fork.md](fork.md) lists every static reference to change.

## Asking for the contract to change

Open an issue first. `contract/pipeline.proto` is a wire format shared by every
box and every client in the fleet; field numbers are permanent, so additions
are possible and renumbering is not.
