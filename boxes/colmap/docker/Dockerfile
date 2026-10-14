# Define workdir folder for all stages
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
ARG SERVICE_NAME=colmap

# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

# -----------------------------
# Stage to generate final image
# -----------------------------
#
# Base image is the official, already-compiled COLMAP (CUDA build). We do
# NOT rebuild C++ here - that is COLMAP's job; our image only adds the
# Python service on top. The box needs the `colmap` executable and its
# optional GPU bits, all present in that image. Pinning by digest.

FROM colmap/colmap@sha256:c1de51615dff33066cd305071226b6343c1168324ef3bd622f0825e55feb5efe
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

# Python for the gRPC service. The base is Ubuntu with an old-enough glibc
# that manylinux wheels of grpcio cover; a venv keeps newer releases with a
# PEP-668 (externally-managed) pip from breaking the install.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

ENV PATH="/opt/venv/bin:${PATH}" \
    XDG_CACHE_DIR=/workspace/.xcache

COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file (renamed to service.py for generic usage) and protos/
# so the service's `sys.path.append("./protos")` resolves under any cwd.
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py
COPY --chown=${USER}:${GROUP} protos ${WORKSPACE}/protos
# the service resolves the stubs via sys.path.append("./protos") - keep a
# copy there (aux.py/serve.py stay at the workspace root as well)
RUN cp ${WORKSPACE}/pipeline_pb2.py ${WORKSPACE}/pipeline_pb2_grpc.py ${WORKSPACE}/protos/

USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
