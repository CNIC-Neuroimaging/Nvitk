# PESA-FAT Analysis Pipeline Container

Container for PESA-FAT (Progression of Early Subclinical Atherosclerosis - Fat Analysis Tool) analysis pipeline.

## 📋 Overview

This container provides a complete environment for cardiovascular MRI analysis, including tissue segmentation and fat quantification for the PESA-FAT study.

**Type:** Project Container  
**Base:** gpu-base  
**GPU Required:** Yes  
**Minimum GPU Memory:** Unknown

## 🔧 What's Included

### Base Environment
- All components from gpu-base container
- CUDA 12.8, PyTorch, CuPy

### Project-Specific Tools
- TotalSegmentator v2 support
- Fat analysis and quantification tools
- MRI image processing capabilities

## 🚀 Usage

### Finding the Container

```bash
# Get latest version
jq -r '.containers.projects["pesa-fat"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' registry/containers.json
```

### Running the Container

TBA

### Required Bind Mounts

TBA

## 📦 Dependencies

### Base Container
- **gpu-base:** v2025.5.13

### Models
- **TotalSegmentator v2:** Required for tissue segmentation
  - Location: `/ia_models/imaging/totalsegmentator/totalsegmentator-v2`
  - Environment variable: `TOTALSEG_HOME_DIR=/models/imaging/totalsegmentator/totalsegmentator-v2`

## 🔨 Building from Source

### Update Requirements

The actual requirements are currently in:
- `/home/imarcoss/BioImaging/env/gpu-pesa_conda.txt`
- `/home/imarcoss/BioImaging/env/gpu-pesa_pip.txt`

### Build Command

```bash
# Auto-generate version
./scripts/build_container.sh projects/pesa-fat

# Specify version
./scripts/build_container.sh projects/pesa-fat v2025.10.13
```

## 📝 Files in this Directory

```
projects/pesa-fat/
├── singularity-gpu-pesa-fat.def.template  # Template definition
├── .container-metadata.yml                # Container metadata
├── README.md                              # This file
├── CHANGELOG.md                           # Version history
└── requirements/
    ├── environment.yml                    # Conda environment spec
    ├── conda_requirements.txt             # Conda packages
    └── pip_requirements.txt               # Pip packages
```

## 🔍 Verification

Verify container integrity:

```bash
./scripts/verify_container.sh projects/pesa-fat v2025.5.27
```

## 📊 Container Metadata

- **Maintainer:** Ignacio Marcos Serrano (imarcoss@cnic.es)
- **Base Container:** gpu-base
- **GPU Required:** Yes
- **Min GPU Memory:** Unknown

## 📚 Related Documentation

- [Container Guide](../../CONTAINER_GUIDE.md)
- [Build Scripts](../../scripts/README.md)
- [Registry](../../registry/README.md)
- [Base Container](../../base/gpu-base/README.md)

## 🐛 Troubleshooting

### Model Not Found

```bash
# Ensure /ia_models is mounted correctly
apptainer exec --nv --bind /ia_models:/models pesa-fat.sif \
    ls /models/imaging/totalsegmentator/

# Check environment variable
apptainer exec --nv --bind /ia_models:/models pesa-fat.sif \
    printenv TOTALSEG_HOME_DIR
```

### GPU Issues

```bash
# Verify GPU access
apptainer exec --nv --bind /ia_models:/models pesa-fat.sif \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📞 Support

For issues or questions:
- Open an issue in the GitLab repository
- Contact: imarcoss@cnic.es
