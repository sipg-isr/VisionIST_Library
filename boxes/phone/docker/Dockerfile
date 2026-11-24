# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
# The python file should have the name ${SERVICE_NAME}_service.py
# All boxes share the same proto file: pipeline.proto
ARG SERVICE_NAME=phone


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


# ----------------------------------------------------------
# Builder stage for the 3D viewer shown on the phone
# ----------------------------------------------------------
# Results can carry a binary glTF (a 3D map). The page shows it with three.js,
# bundled HERE into one file so the phone never fetches a library from a third
# party and the box never reaches the network at request time. three is pinned;
# if this stage is removed the page still works and offers the model for
# download instead.
# TODO: pin node by digest like the python base above (docker buildx imagetools
# inspect node:20-slim) - left as a tag because it could not be resolved when
# this box was written.
FROM node:20-slim AS viewer
ARG THREE_VERSION=0.160.0
ARG ESBUILD_VERSION=0.19.12
WORKDIR /viewer
COPY src/web/viewer.src.js ./viewer.src.js
RUN npm init -y >/dev/null && \
    npm install --no-audit --no-fund three@${THREE_VERSION} esbuild@${ESBUILD_VERSION} && \
    npx esbuild viewer.src.js --bundle --minify --format=esm --outfile=/out/viewer.js


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

# In-container HTTP(S) port of the phone page (extra_ports in box.yaml: the
# fleet generator sets this to the same number it publishes).
ENV WEB_PORT=8062

# Install requirements. Nothing here needs apt: no OpenCV, no CUDA.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file, its modules and the web page to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# The bundled 3D viewer, built in the `viewer` stage.
COPY --from=viewer --chown=${USER}:${GROUP} /out/viewer.js ${WORKSPACE}/web/vendor/viewer.js

# Change to non-privileged user
USER ${USER}

# 8061: gRPC (ai4eu specification). 8062: the phone page (HTTP, or HTTPS when
# TLS_CERT/TLS_KEY are set).
EXPOSE 8061 8062

WORKDIR ${WORKSPACE}


CMD ["python3", "service.py"]
