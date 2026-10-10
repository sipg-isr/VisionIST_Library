# Forking VisionIST_Library into your own registry

This page lists every static reference in the repository that points at the
original owner (GitHub `sipg-isr`, Docker Hub `sipgisr`) and what to do with
it if you want your own registry that the **VisionIST client** can still use.

The client is box-agnostic. It needs a `host:port`, the envelope in
`contract/` and the declared codecs. It does not care who built a box or where
the image lives. So a fork only has to keep the contract intact and repoint the
names below. Everything in section 1 is generated or substituted from the
manifests; section 2 lists the places a human must edit.

Re-run the scan at any time. After you have forked it should print only the
three `pivist/features` attribution lines of the `features` box (see 2.6):

```bash
grep -rnI "sipg-isr\|jpcosteira\|sipgisr\|Costeira\|isr.tecnico\|pivist" . \
  --exclude-dir=.git --exclude=index.json
```

## 1. Quick path

Pick your values once:

```bash
GH_OWNER=youruser            # GitHub account or org that owns the fork
IMG_ORG=yourorg              # Docker Hub namespace for the images
NAME="Your Name"; EMAIL=you@example.org
```

Then, from the repository root:

```bash
# 1. maintainers, owners, image names, URLs, docs
grep -rlI "sipg-isr" . --exclude-dir=.git --exclude=index.json | xargs sed -i "s/sipg-isr/$GH_OWNER/g"
# and the maintainer handle in every box.yaml, which is a person, not the org:
grep -rlI "jpcosteira" . --exclude-dir=.git --exclude=index.json | xargs sed -i "s/jpcosteira/$GH_HANDLE/g"
grep -rlI "sipgisr"    . --exclude-dir=.git --exclude=index.json | xargs sed -i "s#sipgisr/#$IMG_ORG/#g; s#/repositories/sipgisr#/repositories/$IMG_ORG#g; s#\`sipgisr/\`#\`$IMG_ORG/\`#g"
sed -i "s/name: Joao Paulo Costeira/name: $NAME/; s/email: jpc@isr.tecnico.ulisboa.pt/email: $EMAIL/" boxes/*/box.yaml

# 2. regenerate everything derived from the manifests
python tools/build_index.py          # index.json, README table, CODEOWNERS, .gitignore block
python tools/validate_boxes.py
python tools/check_docs.py
```

macOS: use `sed -i ''`. The sed lines above are mechanical; the sections below
explain what each one touches and the few things sed cannot fix.

Keep the image prefix `visionist-` if you want `make_fleet.py`, the READMEs and
`check_docs.py` to stay consistent. You may rename it, but then you must change
every place listed in 2.2.

## 2. Static references and what to do

### 2.1 Owner and maintainers

| Where | Reference | What to do |
|---|---|---|
| `boxes/*/box.yaml`, `maintainers:` | `github: jpcosteira`, name, email | Replace with yours (step 1 above). Required by `registry/schema.json`. |
| `tools/build_index.py` L104-107 | `/contract/ /tools/ /registry/ /.github/` owned by `@jpcosteira` | Change the four owners, rerun `build_index.py`. |
| `.github/CODEOWNERS` | Generated | Do not edit by hand; it is rewritten by `build_index.py`. |
| `tools/new_box.py` | Leaves maintainer name and email as template placeholders | Fill them in the generated `box.yaml` by hand, or patch the script to read them from git config. |
| `template_box/box.yaml` | Placeholders (`your-github-handle`, `you@example.org`) | Nothing to do. |

### 2.2 GitHub and registry URLs

| Where | Reference | What to do |
|---|---|---|
| `tools/make_fleet.py` L26 (`DEFAULT_INDEX_URL`), also its docstring (L17) and the README it emits (L162) | `raw.githubusercontent.com/sipg-isr/VisionIST_Library/main/registry/index.json` | Point at your raw `registry/index.json`. Users can also override per run with `--index <url-or-path>`. For a private repo the URL needs a token, or distribute the file and use `--index`. |
| `VisionIST/tools/make_fleet.py` (client repo, identical copy) | same constant | Change it there too, or always pass `--index`. |
| `registry/schema.json` `$id` | `https://github.com/sipg-isr/VisionIST_Library/registry/schema.json` | Change to your repo; it is only an identifier, nothing fetches it. |
| `README.md` L7, L23, `CONTRIBUTING.md` L9 | Links to `sipg-isr/VisionIST`, `VisionIST_matlab`, `VisionIST_Library`, clone command | Edit the links; point `VisionIST` and `VisionIST_matlab` at your forks only if you forked those too. |
| `README.md` L25 | `hub.docker.com/repositories/sipgisr` | Your Docker Hub page. |

### 2.3 Image, registry and container names

| Where | Reference | What to do |
|---|---|---|
| `boxes/*/box.yaml` `image.repository`, and `template_box/box.yaml` | `sipgisr/visionist-<name-with-hyphens>` | Change the namespace, keep the `visionist-<name>` part. Rerun `build_index.py`, which rewrites `registry/index.json` (it embeds `docker.io/<repo>:<version>`). |
| `tools/new_box.py` L74-75, L90, L92 | prefix `sipgisr/visionist-` for new boxes | Change the namespace. |
| `tools/check_docs.py` (docstring L7-9, L101-102) | Expects README `docker build`/`docker run` to use the manifest repository | No code change needed if manifests and READMEs agree; only the docstring examples mention `sipgisr`. |
| Box READMEs, `docker/README.md`, `QUICKSTART.md`, `README.md` (L19, L29, L31), `CONTRIBUTING.md` L100 | `docker build --tag sipgisr/visionist-<name>`, `docker run ... sipgisr/visionist-<name>` | Covered by the sed in step 1. `check_docs.py` fails if a README and its manifest disagree. |
| `tools/_registry.py` L20-21, L61-63 | Registry host map `ghcr` -> `ghcr.io`, `dockerhub` -> `docker.io`; Docker Hub flattens `org/name` | Docker Hub: nothing. Another registry (Quay, a private host): add it to `REGISTRY_HOSTS`, to the `enum` in `registry/schema.json` (L50), to `--registry` choices in `make_fleet.py` (L203) and list it under each box's `registries:`. |
| `tools/make_fleet.py` L99, L113, L131 | Compose project `visionist-fleet`, containers `visionist-<box>`, `visionist-webui`; base port 9061; webui 8080 -> 8000 | Optional. Rename only if you need to run two fleets side by side. Changing the ports is also safe for the client. |
| `.github/workflows/validate.yml` | `box-under-test`, `--name box` | CI-local names. Leave as they are. |
| `CONTRIBUTING.md`, box READMEs | `--name my_box` and similar | Examples only. |

### 2.4 CI and publishing

| Where | Reference | What to do |
|---|---|---|
| `.github/workflows/publish-box.yml` | Secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`; optional `ghcr.io` login with the workflow token | In your repo, Settings -> Secrets and variables -> Actions, add both secrets for your Docker Hub account, which must own `$IMG_ORG`. Tag `<box>-v<version>` to publish. For GHCR, add `ghcr` to the box's `registries:`; no secret is needed. |
| `.github/workflows/validate.yml` | Runs static checks, the changed-box matrix and conformance | No change. It reads boxes from the manifests. |

### 2.5 Fixtures and assets

The Library no longer fetches fixtures from any other repository. Every box's
test images are committed in its own `test/` folder:

| Box | Fixture |
|---|---|
| clip, open_clip, yolo, lang_sam, moge | `test/car.jpg` (one photo, same bytes) |
| features, lightglue, opencv | `test/00.jpg`, `test/01.jpg` |
| unimatch | `test/flow_0.jpg`, `test/flow_1.jpg` |
| tapnext | `test/pan.mp4`, generated from `lightglue/test/00.jpg` by `test/make_test_video.py` |

The `assets:` mechanism (`tools/fetch_assets.py`, sha256 in `box.yaml`) is still
supported for files over the 256 KB fixture cap (`MAX_FIXTURE_BYTES` in
`tools/_registry.py`). If you add one, host it in your own repo or bucket and
record the sha256. `yolo/test/test_yolo.py` borrows `tapnext/test/pan.mp4` for
its optional video case; set `YOLO_TEST_VIDEO` to use another file.

### 2.6 Upstream sources baked into the images

These are third-party projects. A fork can keep them, but they are the part
most likely to break a rebuild, so decide whether to mirror or pin them.

| Box | Reference | Note |
|---|---|---|
| d4rt | `Lijiaxin0111/Open-d4rt` at `D4RT_COMMIT`; weights from HF `Lijiaxin0111/OpenD4RT` | pinned by commit |
| lang_sam | `luca-medeiros/lang-segment-anything` (`LANGSAM_REPO`) | unpinned |
| lightglue, opencv | `pip install git+https://github.com/cvg/LightGlue.git` | unpinned |
| tapnext | `google-deepmind/tapnet`; checkpoint on `storage.googleapis.com/dm-tapnet` | unpinned |
| unimatch | `autonomousvision/unimatch` at `UNIMATCH_COMMIT`; weights on `s3.eu-central-1.amazonaws.com/avg-projects/unimatch` | pinned by commit |
| vggt | `facebookresearch/vggt@main`; weights HF `facebook/VGGT-1B` | code unpinned |
| moge | `microsoft/MoGe@main`; extra index `download.pytorch.org/whl/cu122` | unpinned |
| clip | `openai/CLIP` (git requirement) | unpinned |
| features | credits `pivist/features` (the SIFT CLI that was ported) in `box.yaml`, README and service docstring | attribution only; nothing is fetched |

Base images are digest-pinned where the Dockerfile shows `@sha256:`
(`python:slim-buster`, `python:3.10-slim`, `nvidia/cuda:12.2.2-*`); d4rt uses
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, and the moge and vggt build
stages are not pinned. Update the digests when you rebuild for security fixes.

### 2.7 Paths and variables inside the images

Internal to each image; a fork does not need to change them.

- Every service appends `./protos` to `sys.path`, listens on `PORT` (default 8061) and runs with the box directory as working directory. The Dockerfiles copy `src/`, `protos/` and `requirements.txt` into `/workspace` (`WORKSPACE`).
- Caches and weights: `/workspace/.cache`, `.torch_cache`, `.xcache`, `/weights` (d4rt, vggt), `/opt/opend4rt`, `/workspace/bootstapnext_ckpt.npz`; `TORCH_HOME`, `HF_HOME`, `HF_HUB_CACHE`.
- Per-box variables: `SERVICE_NAME` (build arg, must equal the box key), `OPEN_CLIP_MODEL`, `OPEN_CLIP_PRETRAINED`, `D4RT_*`, `YOLO_WEIGHTS`, `UNIMATCH_*`, and the `*_SESSION_TTL` timeouts.
- `lang_sam` still accepts the legacy config sections `lang_segm` and `aispgradio` next to `lang_sam`.

### 2.8 Layout constants used by the tools

`boxes/`, `template_box/`, `contract/`, `registry/index.json`, the
`<!-- BEGIN BOXES -->` markers in `README.md`, the `BEGIN ASSETS` block in
`.gitignore`, and `GLOBAL_PATHS` / `MAX_FIXTURE_BYTES` in `tools/_registry.py`.
Keep them as they are; the generators and CI depend on them.

## 3. What must not change if you want VisionIST client compatibility

- `contract/pipeline.proto` and `aux.py`: field numbers are permanent. Boxes vendor them with `tools/sync_contract.py`.
- The config convention: the request's `config_json` section is named after the box key, and `command` `reset` is supported.
- The declared codec names: `identity`, `json`, `numpy`, `torch`, `zstd_pickle`, and the statuses `done`, `empty_request`, `error`.
- Port 8061 inside every container.

Box keys, directory names, owners, image namespaces, registries and the
`visionist-` prefix may all differ. After the changes, check a fork end to
end with:

```bash
python tools/make_fleet.py --index registry/index.json --boxes clip --out /tmp/myfleet
docker compose -f /tmp/myfleet/docker-compose.yml up -d
python contract/conformance/test_conformance.py --box clip --host localhost:9061
```

The fleet pulls `docker.io/<your org>/visionist-clip:<version>`, so the image
must already be published (tag `clip-v<version>`).
