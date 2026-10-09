# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
# The python file should have the name ${SERVICE_NAME}_service.py
# All boxes share the same proto file: pipeline.proto
ARG SERVICE_NAME=pycv


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=.  --grpc_python_out=. pipeline.proto


# -----------------------------
# Stage to generate final image
# -----------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src


# Create non-privileged user and workspace
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

ENV XDG_CACHE_DIR=/workspace/.xcache

# Install requirements
# libgl1 / libglib2.0-0 are OpenCV's runtime shared libraries. Nothing CUDA,
# nothing deep: this box is numpy + OpenCV and little else, which is what
# keeps it around 400 MB rather than the 4 GB of the `opencv` box.
COPY requirements.txt .
RUN apt update -y && \
    apt install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
# src/ also carries vs_prelude.py, which the service copies into each run
# directory as vs.py - that is what a user's script imports.
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}


CMD ["python3", "service.py"]

# This box runs code it is sent. The image cannot make that safe on its own -
# see the README for the compose settings that bound it (read_only, tmpfs,
# cap_drop, mem_limit, pids_limit) and for why the port belongs on a trusted
# network only.
