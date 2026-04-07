# Changelog - eICAB Container

All notable changes to the eICAB (Express Intracranial Arteries Breakdown) container are documented in this file.

**Note:** This is an external container. We do not have access to the source `.def` files.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and versions use [Calendar Versioning](https://calver.org/) (YYYY.M.D).

---

## [v2023.4.17] - 2023-04-17

### Fixed
- ✅ **ANTs dependencies** - Fixed broken dependencies for [ANTs](https://github.com/ANTsX/ANTs.git) (Advanced Normalization Tools)
  - ANTs is required for image registration and brain segmentation
  - Original container had incomplete or broken ANTs installation
  - Updated to working ANTs version compatible with eICAB workflows

- ✅ **AFNI dependencies** - Fixed broken dependencies for [AFNI](https://afni.nimh.nih.gov/) (Analysis of Functional NeuroImages)
  - AFNI is required for functional neuroimaging analysis
  - Original container had missing or incompatible AFNI components
  - Updated to working AFNI version

### Changed
- Verified all core dependencies are properly installed and functional
- Tested Circle of Willis segmentation

### Container Details
- **Source:** Modified external container from 
- **Base:** Unknown (external build)
- **GPU Required:** No
- **Primary Use:** Circle of Willis instance segmentation from MRA 

### Notes
This version resolves critical dependency issues that prevented the original container from functioning properly. All essential tools (eICAB, ANTs, AFNI) are now fully operational.

**Recommended:** Use this version for all new analyses.

---

## [v2022.10.15] - 2022-10-15

### Added
- ✅ Initial version - Direct download from [eICAB repository](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab)

### Known Issues
- ❌ **Broken ANTs dependencies** - ANTs tools not fully functional
- ❌ **Broken AFNI dependencies** - AFNI tools not fully functional

### Container Details
- **Source:** Direct download from [eICAB repository](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab)
- **Base:** Unknown (external build)
- **GPU Required:** No
- **Status:** Deprecated - Use v2023.4.17 instead

### Notes
This is the original container as distributed by the eICAB project. Due to dependency issues, it is **not recommended** for production use. Please use v2023.4.17 or later.

**Deprecation Notice:** This version is kept for historical reference only.

---

## Version Comparison

| Feature | v2022.10.15 | v2023.4.17 |
|---------|-------------|------------|
| **eICAB Core** | ❌ Broken | ✅ Working |
| **ANTs Integration** | ❌ Broken | ✅ Fixed |
| **AFNI Integration** | ❌ Broken | ✅ Fixed |
| **Production Ready** | ❌ No | ✅ Yes |
| **Recommended** | ❌ No | ✅ Yes |

---

## Dependency Information

### ANTs (Advanced Normalization Tools)
- **Repository:** https://github.com/ANTsX/ANTs.git
- **Purpose:** Image registration, segmentation, and normalization
- **Key Tools:** antsRegistration, antsBrainExtraction, Atropos
- **Status in v2023.4.17:** ✅ Fully functional

### AFNI (Analysis of Functional NeuroImages)
- **Website:** https://afni.nimh.nih.gov/
- **Purpose:** Functional neuroimaging analysis
- **Key Tools:** 3dSkullStrip, 3dAllineate, 3dvolreg
- **Status in v2023.4.17:** ✅ Fully functional

### eICAB
- **Purpose:** Express Intracranial Arteries Breakdown
- **Features:** 
  - Automated segmentation and labeling of the main intracranial arteries on MR angiography images
- **Status:** ✅ Working in last version

---

## Migration Guide

### From v2022.10.15 to v2023.4.17

**How to upgrade:**

Update your scripts to point to the new version:
```bash
# Old (v2022.10.15)
CONTAINER="/containers/projects/neuroimaging/eICAB/eICAB_v2022.10.15.sif"

# New (v2023.4.17) - Recommended
CONTAINER="/containers/projects/neuroimaging/eICAB/eICAB_v2023.4.17.sif"
```
---

## Support and Issues

For issues specific to:

- **Container deployment:** Contact imarcoss@cnic.es or open issue in BioImaging-Containers
- **eICAB methodology:** Refer to [eICAB project documentation](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab)
- **ANTs tools:** See [ANTs documentation](https://github.com/ANTsX/ANTs)
- **AFNI tools:** See [AFNI documentation](https://afni.nimh.nih.gov/)

---

## References

### eICAB
- Original eICAB repository and documentation [GitLab](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab)

### ANTs
- GitHub: https://github.com/ANTsX/ANTs.git

### AFNI
- Website: https://afni.nimh.nih.gov/

---

## Maintainer Notes

**External Container Status:**
- ✅ Metadata documented
- ✅ Usage tested
- ✅ Dependency issues resolved (v2023.4.17)
- ❌ Source `.def` not available
- ❌ Cannot rebuild from source

**Maintenance:**
- Container is used as-is
- No further modifications planned
- Consider requesting source from eICAB project for future updates

---

**Last Updated:** October 13, 2025  
**Maintainer:** BioImaging Team (djimenez@cnic.es)  
**Latest Stable Version:** v2023.4.17

