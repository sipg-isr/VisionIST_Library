# internvl box - InternVL3 visual question answering.
#
# Two differences from the fleet's standard template, both forced by the model:
#
#   * the base image is PyTorch's CUDA 12.8 runtime rather than python:slim.
#     CUDA 12.8 / torch 2.7 is the first combination with kernels for
#     Blackwell (RTX 50xx, B200, sm_120/sm_100) as well as Turing through
#     Hopper, so one image serves every NVIDIA GPU from an RTX 2080 up. It
#     also runs, slowly, on CPU;
#   * the checkpoint is baked in at build time (docker/fetch_weights.py), so a
#     fleet never waits on a Hugging Face download and the box never reaches
#     the network at request time. InternVL3-1B is ~2 GB. Pick another size,
#     or a fine-tuned checkpoint on the Hub, with --build-arg MODEL_ID=...;
#     pass --build-arg DOWNLOAD_WEIGHTS=false to build a thin image and mount
#     a checkpoint at /weights/model instead.
#
# Build:
#   cd boxes/internvl
#   docker build --tag sipgisr/visionist-internvl --build-arg SERVICE_NAME=internvl -f docker/Dockerfile .
#
#   # a bigger model, pinned to a commit
#   docker build --tag sipgisr/visionist-internvl --build-arg SERVICE_NAME=internvl --build-arg MODEL_ID=OpenGVLab/InternVL3-8B --build-arg MODEL_REVISION=<sha> -f docker/Dockerfile .
#
# flash-attn is NOT installed by default: the model's eager attention works on
# every GPU and on CPU. --build-arg INSTALL_FLASH_ATTN=true adds it (Ampere or
# newer only; pip fetches a prebuilt wheel when one matches this torch/CUDA/
# Python, and otherwise compiles, which needs a -devel base and a long build).
# Then run with -e INTERNVL_FLASH_ATTN=1.

ARG WORKSPACE=/workspace
ARG SERVICE_NAME=internvl


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
ARG WORKSPACE
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto


# ------------------------------------------
# Builder stage to fetch the model checkpoint
# ------------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS weights
ARG MODEL_ID=OpenGVLab/InternVL3-1B
ARG MODEL_REVISION=main
ARG DOWNLOAD_WEIGHTS=true
ENV MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION} \
    DOWNLOAD_WEIGHTS=${DOWNLOAD_WEIGHTS}

RUN pip install --no-cache-dir "huggingface_hub>=0.24"
COPY docker/fetch_weights.py /tmp/fetch_weights.py
RUN python /tmp/fetch_weights.py


# -----------------------------
# Stage to generate final image
# -----------------------------
# Pin by digest once the image is first built: docker inspect --format
# '{{index .RepoDigests 0}}' pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ARG WORKSPACE
ARG SERVICE_NAME
ARG INSTALL_FLASH_ATTN=false

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INTERNVL_MODEL_DIR=/weights/model \
    HF_HOME=/workspace/.hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# torch comes from the base image; requirements.txt holds everything else.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt
RUN if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Non-privileged user, created before the weights are copied so that
# COPY --chown sets ownership in the same layer (a later chown -R would store
# the multi-GB checkpoint twice).
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir -p ${WORKSPACE}/.hf && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

COPY --from=weights --chown=${USER}:${GROUP} /weights /weights

# The .proto at the root meets the AI4EU spec.
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

USER ${USER}
EXPOSE 8061
WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
