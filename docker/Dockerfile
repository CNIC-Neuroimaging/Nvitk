# syntax=docker/dockerfile:1
# NvITK 1.0.0 base container
# ----- Base Image: CUDA 13.2 + cuDNN -----
ARG BASE_IMAGE=nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04
FROM --platform=linux/amd64 ${BASE_IMAGE}

# nvitk version from the cnic channel
ARG NVITK_VERSION=1.0.0
# gpu: torch cu130 + cupy/cutensor/nccl | cpu: torch CPU, no CUDA packages
ARG NVITK_VARIANT=gpu
# 1 = FastSurfer (tree + checkpoints) in the image
ARG INSTALL_FASTSURFER=1
# 1 = nvitk's own FSL core in the image (nvitk-fsl install); 0 = mount an FSL and set FSLDIR
ARG INSTALL_FSL=1

LABEL org.opencontainers.image.source="https://github.com/ignacio-ms/Nvitk" \
        org.opencontainers.image.version="1.0.0" \
        org.opencontainers.image.description="NvITK 1.0.0 | QVTPy"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# ----- System packages -----
# curl/bzip2/ca-certificates: Miniforge, and the VMTK wheel and FastSurfer downloads in post-link
# build-essential/git: sdists built by post-link (pyradiomics, the VMTK fallback build)
# GL/EGL/X11/xcb/fontconfig/dbus: VTK, OpenCV and Qt (napari, offscreen rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget bzip2 git build-essential procps \
        libgl1 libegl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
        libfontconfig1 libdbus-1-3 libxkbcommon0 libxkbcommon-x11-0 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

# ----- Conda (Miniforge: conda + mamba, conda-forge by default) -----
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda clean -a -y

# ----- Install NvITK -----
# Its own env, not base: nvitk pins python 3.11.15 and base runs conda itself.
# post-link.sh installs torch, VMTK, etc. with pip; FastSurfer goes in its own layer below
# (NVITK_FASTSURFER=0 here) so no single layer exceeds GHCR's 10 GB limit.
RUN NVITK_VARIANT=${NVITK_VARIANT} NVITK_FASTSURFER=0 \
        conda create -y -n nvitk -c cnic -c conda-forge nvitk=${NVITK_VERSION} \
    && conda clean -a -y \
    && rm -rf /root/.cache/pip

ENV CONDA_DEFAULT_ENV=nvitk \
    CONDA_PREFIX=${CONDA_DIR}/envs/nvitk \
    PATH=${CONDA_DIR}/envs/nvitk/bin:${CONDA_DIR}/bin:${PATH} \
    QT_PLUGIN_PATH=${CONDA_DIR}/envs/nvitk/lib/qt6/plugins

# ----- FastSurfer (optional) -----
RUN if [ "${INSTALL_FASTSURFER}" = "1" ]; then nvitk-fastsurfer install; fi \
    && rm -rf /root/.cache/pip

# ----- FSL core (optional) -----
# --cuda explicitly: there is no nvidia-smi at build time, so auto-detection would choose CPU.
RUN if [ "${INSTALL_FSL}" = "1" ]; then \
        if [ "${NVITK_VARIANT}" = "gpu" ]; then CUDA_FLAG=--cuda; else CUDA_FLAG=--no-cuda; fi; \
        nvitk-fsl install -y ${CUDA_FLAG} && conda clean -a -y; \
    fi

# ----- Entrypoint: full activation (conda activate.d hooks: LD_LIBRARY_PATH, Qt, FastSurfer, R) -----
RUN printf '%s\n' '#!/bin/bash' \
        "source ${CONDA_DIR}/etc/profile.d/conda.sh" \
        'conda activate nvitk' \
        'exec "$@"' > /usr/local/bin/nvitk-entrypoint \
    && chmod +x /usr/local/bin/nvitk-entrypoint

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/nvitk-entrypoint"]
CMD ["bash"]