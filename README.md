# VisionIST Library


This is a companion repository of SIPG/ISR's [VisionIST](https://github.com/sipg-isr/VisionIST). Together with [VisionIST_matlab](https://github.com/sipg-isr/VisionIST_matlab) allows the local buildup of any type of processing "pipeline" running components in a distributed way. Each component follows the specifications of the  ["AI-on-Demand"](http://aiod.eu) platform. We call such components **boxes** or **caixinhas**.

**VisionIST_Library** is the registry of **boxes**: independent, Dockerized inference services that all
speak one gRPC envelope. Contribute a box here; assemble a fleet from it
anywhere.

```protobuf
service PipelineService { rpc Process( Envelope ) returns ( Envelope ); }
```

This repository holds **recipes and metadata** — a Dockerfile, the service source, a manifest. The **images** live in a container registry (Docker Hub,
`sipgisr/`). So running a fleet never means building a hundred boxes, and cloning the registry is not a prerequisite for using it.

The client, the webui, the docs and a reference fleet live in
[VisionIST](https://github.com/sipg-isr/VisionIST) and we make also a  [VisionIST_Matlab](https://github.com/sipg-isr/VisionIST_matlab) client.

Visit our  [Docker Hub repositories](https://hub.docker.com/repositories/sipgisr). VisionIST componentes are named sipgisr/visionist-name-of-the-box

Want your own registry that the VisionIST client can still use? See [fork.md](fork.md).

## Use a box

```bash
docker run --rm -p 8061:8061 -e PORT=8061 docker.io/sipgisr/visionist-clip
# or, from Docker Hub:
docker run --rm -p 8061:8061 -e PORT=8061 docker.io/sipgisr/visionist-clip
```

**Call a box with the VisionIST** client (see each box definitions to tailor the client call) 

```python
from visionist_client import Visionist
b = Visionist("localhost:8061")
res = b.run(data={"images": [pathlib.Path("car.jpg")], "texts": ["a race car"]},
            config={"clip": {}})
```

## Build a fleet

```bash
python3 tools/make_fleet.py --boxes clip,yolo,moge --out my-fleet --gpu
cd my-fleet && docker compose pull && docker compose up -d
```

Host ports are assigned automatically, the compose file pulls rather than
builds, and `--registry dockerhub` switches where from. You do not need this
checkout: `--index` also takes the published
[`registry/index.json`](registry/index.json) by URL.

## The boxes

<!-- BEGIN BOXES -->
| Box | Key | Runtime | Port | What it does | Tags |
|---|---|---|---|---|---|
| [`clip`](boxes/clip) | `clip` | gpu | 9061 | CLIP image and text embeddings with their cross-modal similarity | `embeddings` `multimodal` |
| [`d4rt`](boxes/d4rt) | `d4rt` | gpu | 9072 | OpenD4RT 4D reconstruction and tracking - 3D point tracks, point clouds and camera poses from a video | `depth` `dynamic-3d` `geometry` `reconstruction` `tracking` |
| [`features`](boxes/features) | `features` | cpu | 9071 | SIFT keypoints in the SIFT-Extractor layout: (2+128) x N, annotated JPEGs, MATLAB .mat | `classical` `features` `sift` |
| [`lang_sam`](boxes/lang_sam) | `lang_sam` | gpu | 9064 | Text-guided segmentation with LangSAM: phrases in, masks out | `grounding` `language` `segmentation` |
| [`lightglue`](boxes/lightglue) | `lightglue` | gpu-or-cpu | 9069 | SuperPoint/DISK features and LightGlue matching, pairwise or as a sliding-window stream | `features` `matching` `tracking` |
| [`moge`](boxes/moge) | `moge` | cuda-only | 9067 | MoGe-3 monocular geometry: metric depth, point map, normals, intrinsics | `depth` `geometry` `monocular` |
| [`open_clip`](boxes/open_clip) | `open_clip` | gpu-or-cpu | 9073 | OpenCLIP image and text embeddings from any open_clip model and pretrained tag, with cosine similarity and zero-shot probabilities | `clip` `embeddings` `multimodal` `zero-shot` |
| [`opencv`](boxes/opencv) | `opencv` | gpu-or-cpu | 9065 | Classic feature extraction and matching: SIFT/ORB via FLANN, or SuperPoint/DISK via LightGlue, plus a RANSAC fundamental matrix | `classical` `features` `matching` |
| [`pycv`](boxes/pycv) | `pycv` | cpu | 9075 | Runs user-supplied Python against OpenCV - a script or a one-line expression in, named arrays out | `classical` `opencv` `scripting` |
| [`sbert`](boxes/sbert) | `sbert` | gpu-or-cpu | 9062 | Sentence-BERT text embeddings and their pairwise similarity | `embeddings` `text` |
| [`sfm`](boxes/sfm) | `sfm` | cpu | 9074 | SfM: camera poses + 3D points from feature tracks and monocular depth, with partial (missing) tracks | `depth` `geometry` `reconstruction` `sfm` |
| [`tapnext`](boxes/tapnext) | `tapnext` | gpu | 9063 | TAPNext point tracking with the Tomasi-Kanade observation matrix | `points` `sfm` `tracking` |
| [`unimatch`](boxes/unimatch) | `unimatch` | gpu-or-cpu | 9070 | Unified dense matching: optical flow, stereo disparity, and multi-view depth | `depth` `flow` `geometry` `stereo` |
| [`vggt`](boxes/vggt) | `vggt` | gpu | 9066 | VGGT multi-view 3D reconstruction: world points, per-view depth, camera poses, and a GLB | `geometry` `multiview` `reconstruction` |
| [`yolo`](boxes/yolo) | `yolo` | gpu-or-cpu | 9068 | YOLO object detection with always-on tracking, on images or video | `detection` `tracking` |
<!-- END BOXES -->

`runtime` reads: `cpu` runs anywhere · `gpu-or-cpu` prefers CUDA but works
without · `gpu` wants one · `cuda-only` refuses to run without one.

`Port` is the box's permanent host port in a fleet, issued once in the order
boxes arrived and kept in [`registry/ports.json`](registry/ports.json). A new
box takes the next port after the highest ever issued, so no existing box ever
moves and a box answers on the same port in every fleet anyone generates.

## Layout

```
contract/            pipeline.proto + aux.py - the single source of truth
  conformance/       the test every box must pass
boxes/<name>/
  box.yaml           the registry entry (schema: registry/schema.json)
  docker/Dockerfile
  src/<key>_service.py
  protos/            GENERATED from contract/ - do not edit
  test/
  README.md          the authoritative request shape for that box
registry/
  index.json         GENERATED from every box.yaml
  ports.json         the host-port ledger - APPEND-ONLY, one port per box
  schema.json        what a valid manifest is
template_box/        copy this to start a box
tools/               sync, validate, index, fetch assets, make a fleet
```

## Contributing a box

```bash
python3 tools/new_box.py my_box --key my_box --summary "What it does"
# write src/my_box_service.py, fill in box.yaml, add a small test fixture
python3 tools/validate_boxes.py my_box
python3 tools/build_index.py
```

Then open a PR. CI validates the manifest, checks the contract has not
drifted, enforces the fixture size cap, and builds only your box.
[CONTRIBUTING.md](CONTRIBUTING.md) has the detail.

## Three rules that keep this scalable

**The contract is generated, never copied by hand.** Each box carries
`protos/` so its directory is self-contained, but `tools/sync_contract.py`
writes it and CI fails a PR where a copy has drifted. The repository this
registry was split out of had three variants of `pipeline.proto` and two of
`aux.py` in circulation at 11 boxes — and the `aux.py` fork was functional,
not cosmetic: one version accepted integer payloads and the other raised, so
whether `data={"n": 3}` worked depended on which box you called.

**Nothing downstream of a manifest is maintained by hand.** The index, the
README table above, `.gitignore`, `CODEOWNERS` and the CI build matrix are all
generated by `tools/build_index.py`. The old repo's publish workflow had a
hand-written matrix; at 11 boxes it listed 8, and three boxes were silently
never built.

**Big fixtures are declared, not committed.** Files over 256 KB go in the
manifest's `assets:` with a URL and a checksum, and `tools/fetch_assets.py`
pulls them at test time. Two PNGs in one box were 11 MB of a 16 MB tree, and
the same 388 KB image was committed three times under three names. Git keeps
every byte of every version forever, for everyone who clones.

## Licence

Per box — see each box's manifest `links.license` and its directory.
