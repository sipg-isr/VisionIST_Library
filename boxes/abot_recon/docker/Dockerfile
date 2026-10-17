# abot_recon box - ABot-Recon streaming 3D reconstruction.
#
# Base image is PyTorch's CUDA runtime rather than python:slim, because the
# upstream pyproject pins torch 2.5.1 and the model is ~1.0 B parameters -
# CPU inference runs but is not practical. The stock build matches upstream's
# released configuration (PyTorch 2.5.1 + CUDA 12.x).
#
# The checkpoint (4.0 GB, Hugging Face acvlab/ABot-Recon) is baked in at
# build time, so the image is self-contained and a fleet never waits on a
# download. Pass --build-arg DOWNLOAD_WEIGHTS=false to build a thin image
# and mount the weights at /weights instead.
#
# Optional, both off by default:
#   --build-arg LOOP_ASSETS=true
#       additionally fetch the ~450 MB of loop-closure retrieval assets
#       (DINOv2-SALAD + DINOv2) into /weights/loop, which is what
#       `loop_closure: true` in a request needs. The code for it is always
#       installed (the [loop] extras), so mounted assets work too.
#   ABOT_RECON_ATTENTION_BACKEND / ABOT_RECON_AMP_DTYPE at launch (env)
#       the `paged` backend needs flashinfer-python, which this image does
#       not carry; `auto` (the default) falls back to PyTorch SDPA, which
#       is upstream's documented no-dependency path.
#
# Build:
#   cd boxes/abot_recon
#   docker build --tag sipgisr/visionist-abot-recon --build-arg SERVICE_NAME=abot_recon -f docker/Dockerfile .
#
# Upstream is pinned by commit: the box uses the documented public API
# (ABotRecon.from_pretrained / .infer), but that API is young, so re-test
# the box when you move the pin.

ARG WORKSPACE=/workspace
ARG SERVICE_NAME=abot_recon

# --------------------------------------
# Builder stage: generate the proto stubs
# --------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
ARG WORKSPACE
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto


# --------------------------------------
# Weights stage: fetch the checkpoint once, in a layer of its own
# --------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS weights
ARG HF_REPO_ID=acvlab/ABot-Recon
ARG DOWNLOAD_WEIGHTS=true
ARG LOOP_ASSETS=false

ENV HF_REPO_ID=${HF_REPO_ID} \
    WANT_WEIGHTS=${DOWNLOAD_WEIGHTS} \
    WANT_LOOP=${LOOP_ASSETS}

RUN pip install --no-cache-dir huggingface_hub
COPY docker/fetch_weights.py /tmp/fetch_weights.py
RUN python /tmp/fetch_weights.py


# --------------------------------------
# Runtime
# --------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG WORKSPACE
ARG SERVICE_NAME
# Pinned upstream commit - see the note at the top of this file.
ARG ABOT_RECON_COMMIT=6e3b2e1257464a74dfceff22eb33190bd228276e
ARG ABOT_RECON_REPO_URL=https://github.com/amap-cvlab/ABot-Recon.git

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ABOT_RECON_CKPT=/weights/abot_recon.safetensors \
    ABOT_RECON_LOOP_DIR=/weights/loop \
    HF_HOME=/workspace/.hf \
    TORCH_HOME=/workspace/.torch_cache

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The upstream checkout. Its pyproject pins torch 2.5.1 / torchvision 0.20.1,
# which this base image already provides, so only the rest is installed.
# The [loop] extras (faiss-cpu, pypose, scipy) are part of the released
# package API, so they come along; the loop *assets* remain optional.
RUN git clone --filter=blob:none ${ABOT_RECON_REPO_URL} /opt/abot-recon && \
    cd /opt/abot-recon && git checkout --quiet ${ABOT_RECON_COMMIT} && \
    rm -rf /opt/abot-recon/.git && \
    pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "/opt/abot-recon[loop]" && \
    rm requirements.txt

COPY --from=weights /weights /weights

# Non-privileged user, after the big layers so they stay cacheable.
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir -p ${WORKSPACE} ${WORKSPACE}/.hf ${WORKSPACE}/.torch_cache && \
    chown -R ${USER}:${GROUP} ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} /weights

# The .proto at the root meets the AI4EU spec.
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

USER ${USER}
EXPOSE 8061
WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
