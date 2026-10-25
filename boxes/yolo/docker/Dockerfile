# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
# The python file should have the name ${SERVICE_NAME}_service.py
# All boxes share the same proto file: pipeline.proto
ARG SERVICE_NAME=yolo


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:slim-buster@sha256:c46b0ae5728c2247b99903098ade3176a58e274d9c7d2efeaaab3e0621a53935 AS builder
## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile the shared proto file
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto


# -----------------------------
# Stage to generate final image
# -----------------------------

FROM nvidia/cuda:12.2.2-base-ubuntu22.04@sha256:1a8a738e81d4adbef0c709241f5238cec5bb77186dcb5b2103db293315ed42d1
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

# Install requirements
COPY requirements.txt .
RUN apt update -y && \
    apt install -y pip git &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

# ultralytics depends on non-headless opencv-python (wants libGL, absent in
# this base image); force the headless build in so it wins the cv2 import.
RUN pip install --force-reinstall --no-deps "opencv-python-headless>=4.10,<5"

# COPY the shared .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# No baked weights on purpose: the box downloads its checkpoint at runtime
# (box startup / first use of parameters.weights) into the container
# workspace (its CWD, owned by the runner user). Keeps the build offline-
# friendly and the image small; switch the default with -e YOLO_WEIGHTS=….

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

# $HOME and weight/cache dirs: ultralytics downloads checkpoints and caches
# under $HOME and the CWD — both must be writable by the non-root runner.
# ($HOME points at the workspace because the runner user has no home dir.)
ENV HOME=${WORKSPACE}
ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache

CMD ["python3", "service.py"]
