# Changelog - GPU Base Container

All notable changes to this container will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and versions use [Calendar Versioning](https://calver.org/) (YYYY.M.D).

## [Unreleased]

## [v2025.10.15] - 2025-10-15

### Changed
- Updated CUDA toolkit from 12.8 to 12.9
- Updated NCCL from 2.25.1 to 2.26.5
- Updated PyTorch to version 2.8.0 with CUDA 12.9 support
- Updated torchvision to version 0.23.0 with CUDA 12.9 support
- Updated CuPy to version 13.6.0
- Updated SciPy to version 1.15.3
- Updated libcusparse from 12-8 to 12-9

### Container Details
- Base image: condaforge/miniforge3
- GPU support: Required
- CUDA version: 12.9

## [v2025.5.13] - 2025-05-13

### Added
- Initial version of GPU base container
- Ubuntu 22.04 + Miniforge3
- Python 3.11
- CUDA 12.8 toolkit
- cuDNN 9
- cuTENSOR 2.2.0
- NCCL 2.25.1
- PyTorch with CUDA 12.8 support
- CuPy for GPU-accelerated computing
- SciPy for scientific computing
- Build metadata recording

### Container Details
- Base image: condaforge/miniforge3
- GPU support: Required
- CUDA version: 12.8

### Notes
- This is the foundational container for all Torch/CuPy GPU-enabled project containers
- Designed for deep learning and GPU-accelerated scientific computing

