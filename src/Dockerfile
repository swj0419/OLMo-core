# NOTE: make sure CUDA_VERSION and TORCH_CUDA_VERSION always match, except for punctuation
ARG CUDA_VERSION="12.4"
ARG TORCH_CUDA_VERSION="124"
ARG TORCH_VERSION="2.6.0
ARG BASE=runtime

#########################################################################
# Build image
#########################################################################

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel as build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja

# Build grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_VERSION="grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@main"
RUN pip wheel --no-build-isolation --no-cache-dir "${GROUPED_GEMM_VERSION}"

# Build flash-attn.
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN wget ${FLASH_ATTN_WHEEL}

# Only keep the target wheels and dependencies with CUDA extensions.
RUN echo "Built wheels:" && ls -lh .
    # && ls -1 | grep -Ev 'grouped_gemm|flash_attn' | xargs rm \
    # && echo "Final wheels:" \
    # && ls -lh .

#########################################################################
# Stable image
#########################################################################

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-${BASE} as stable

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ENV MOFED_VER="24.01-0.3.3.1"
ENV OS_VER="ubuntu22.04"
ENV PLATFORM="x86_64"
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging

# Install torchao.
ARG TORCH_CUDA_VERSION
ARG TORCHAO_VERSION="0.9.0"
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION} \
    torchao==${TORCHAO_VERSION}

# Copy and install wheels from build image.
COPY --from=build /app/build /app/build
RUN pip install --no-cache-dir /app/build/*

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip install --no-cache-dir ring-flash-attn==0.1.4 liger-kernel==0.5.4 transformers && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

LABEL org.opencontainers.image.source https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core

#########################################################################
# Nightly image
#########################################################################

FROM stable as nightly

ARG TORCH_CUDA_VERSION
ARG TORCH_NIGHTLY_VERSION="2.7.0.dev20250202"
RUN pip install --no-cache-dir --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION} \
    torch==${TORCH_NIGHTLY_VERSION}+cu${TORCH_CUDA_VERSION}
