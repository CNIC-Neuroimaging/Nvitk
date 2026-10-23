# Installation

## Install via conda (recommended)

`nvitk` is published on [Anaconda.org](https://anaconda.org/cnic/nvitk) under the `cnic`
channel with the full `nvitk[all]` feature set (GUI, GPU/CUDA 13, and R-based statistics)
bundled in — there is no separate CPU/GPU profile to choose.

<!-- conda config --add channels bioconda
conda config --add channels ejolly
conda config --set channel_priority strict -->
<!-- conda config --add channels mrtrix3 -->

```bash
conda config --add channels cnic
conda config --add channels conda-forge
conda config --set channel_priority strict

conda create -n nvitk-env nvitk
conda activate nvitk-env
```

```{note}
This pulls in the CUDA 13 PyTorch/CuPy stack automatically as part of installation.
Packages with no conda equivalent (e.g. `totalsegmentator`, `nnunetv2`, `antspyx`) are
installed via `pip` through a post-link step — see
[`recipe/post-link.sh`](https://github.com/ignacio-ms/Nvitk/blob/main/recipe/post-link.sh)
for the exact list.
```

```{warning}
Use `conda create`/`conda install`, **not** `mamba create`/`mamba install`. `mamba` does
not run post-link scripts (a known gap across mamba/micromamba), so a mamba-based install
silently skips the entire pip-only dependency stack above — the command succeeds, but
`import nvitk` then fails. Plain `conda` (even with the libmamba solver enabled) works
correctly, since conda's own transaction executor still runs the post-link step regardless
of which solver resolved the environment.
```

```{warning}
**Note:** It is important that the conda-forge channel is on top.
```

Then create a configuration — nvitk ships no site paths of its own, so this is required
before any pipeline will run:

```bash
nvitk-config init        # writes ~/.config/nvitk/{sge,settings,xnat}.json
nvitk-config validate    # shows what still needs filling in
```

See {doc}`configuration` for what each key means and where the files are searched for, then
{ref}`get-the-data` below.

(get-the-data)=
## Configuration and dataset retrieval

nvitk ships **code only**. Site configuration and research data are deliberately kept out of
the package: they are specific to one institution, and the data is not ours to distribute.
How you obtain them depends on whether you have access to CNIC storage.

### If you have access (CNIC)

**Configuration** lives in a private repository. From a clone of nvitk:

```bash
git submodule update --init .nvitk
```

That directory is search location #8, so a clone with the submodule checked out is configured
with nothing further to do. If you installed from conda and have no clone, put it wherever
nvitk looks instead:

```bash
git clone git@github.com:ignacio-ms/nvitk-config.git ~/.config/nvitk
```

**The dataset** is tracked with DVC. The pointer files are public — they are content hashes and
reveal nothing — while the content sits on CNIC storage, so access is enforced by the storage
location rather than by the repository:

```bash
nvitk-dataset status       # what is configured, and what is already present
nvitk-dataset pull         # catalog + measurement tables  (~19 MB)
nvitk-dataset pull --all   # + the prebuilt SQLite index   (~1.3 GB)
```

This works from a plain conda install with no checkout: `dvc get` reads the pointers straight
out of the public repository and fetches the content from the configured remote. It needs the
storage mounted and `dvc` installed (`conda install -c conda-forge dvc`).

The dataset is written to `db.local_fallback_root` (or `db.root` if that is unset). With
neither set, nvitk uses `~/.local/share/nvitk/dataset` for an installed package, or
`<repo>/dataset/nvitk-dataset` inside a source checkout — so a conda install never invents a
repository-shaped directory in your home. `nvitk-dataset status` prints the resolved location
*and which setting produced it*, which is the check worth running before a multi-gigabyte
transfer. It reports each target as `present` (with a file count and size), `empty` or
`missing` — an empty directory left by a scaffold is not mistaken for a completed download —
and fetches land flat: a target is downloaded to a staging path and moved into place, so a
pre-existing directory never ends up holding a nested `tables/tables/` copy.

```{tip}
`--all` is rarely worth it. The SQLite index is derived from the tables and rebuilds locally in
about 15 seconds — far quicker than transferring 1.3 GB:

    python -m nvitk.db.sqlite_index --dataset-root <db.root>
```

`nvitk-dataset pull --rev v0.1.0` pins retrieval to a git tag, giving you the dataset as it was
at a particular release.

The two settings it uses, in `settings.json`:

| Key | Meaning |
|---|---|
| `db.root` | Where the dataset is written |
| `db.dvc_repo` | Repository holding the pointer files (defaults to the public nvitk repo) |
| `db.dvc_remote_url` | Where the content lives. Leave empty to use the repository's own default |

### If you are outside the organisation

You cannot fetch CNIC's configuration or data, and you do not need them — nvitk works against
any dataset in its layout. Three steps:

**1. Create a configuration and fill in your own paths.**

```bash
nvitk-config init
nvitk-config path        # shows which file is in use
```

Only `db.root` in `settings.json` is needed for the library, I/O and image-processing CLIs.
The `sge.json` sections matter only if you run pipelines on a cluster; delete what you do not
use, and nvitk will name any key it needs but cannot find.

**2. Create an empty dataset.**

```python
from nvitk.db.catalog import DatasetCatalog
DatasetCatalog.create_scaffold("~/my-dataset")     # then set db.root to this path
```

This writes the catalog manifests and JSON schemas that define the dataset layout, plus empty
`tables/` and `cache/` directories, and works from an installed package with no checkout.

**3. Leave the DVC settings empty.** `db.dvc_repo` / `db.dvc_remote_url` only apply to CNIC
storage; `nvitk-dataset` is not used outside the organisation.

```{note}
Much of nvitk needs no dataset at all. The image-processing CLIs (`nvitk-morph`,
`nvitk-filter`, `nvitk-restore`, `nvitk-transform`, `nvitk-measure`), the conversion tools and
the Python API all operate on files you pass them. A dataset is only required for the
cohort-level pieces: the pipelines, the {doc}`Stats GUI <stats-gui/index>` and the database
layer.
```

Run {doc}`pyhelp <api/cli-catalog>` after activation for an interactive catalog of every
CLI tool (`pyhelp --no-interactive` for CI/scripts).

## Development / wheel install (`install.sh`)

Contributors working from a clone use `install.sh`, which builds the same environment as the
conda package without going through Anaconda.org: it creates a conda environment from the
recipe's conda-native requirements (`recipe/recipe.yaml`), installs the pip-only stack with the
recipe's own `recipe/post-link.sh`, adds the activation hooks, and finally installs nvitk itself.
A dependency change made in `recipe/` therefore applies to both install routes. It needs
`conda` on `PATH` (e.g. [Miniforge](https://conda-forge.org/download/)).

```bash
git clone https://github.com/ignacio-ms/Nvitk.git && cd Nvitk
./install.sh                    # GPU build (default), editable, env "nvitk-dev"
./install.sh --cpu              # CPU-only build: CPU torch wheels, no CUDA packages
conda activate nvitk-dev
```

| Option | Effect |
|---|---|
| `--gpu` | CUDA 13 `torch`/`torchvision` + `cupy`/`cutensor`/`nccl` (default) |
| `--cpu` | CPU-only `torch`/`torchvision`; no CUDA packages |
| `-n NAME` / `-p DIR` | Target environment (default `nvitk-dev`, or `nvitk` with `--wheel`) |
| `--force` | Remove and recreate the target environment if it already exists |
| `--wheel [FILE]` | Install nvitk non-editably: from `FILE`, or from a wheel built into `dist/` |
| `--dev-tools` / `--no-dev-tools` | Include the `[dev]` extra — pytest, sphinx toolchain (default: on for editable, off for `--wheel`) |
| `--no-fastsurfer` | Skip FastSurfer (tree + network weights, ≈340 MB); `nvitk-fastsurfer install` adds it later |
| `--no-fsl` / `--fsl-components LIST` | Skip nvitk's FSL core, or choose its components (default: structural, registration, diffusion, functional) |
| `--dry-run` | Print the variant, target and conda requirements without installing anything |

Both variants include the GUI and R/statistics stacks; only the PyTorch build differs. The
pip-only step logs to `$CONDA_PREFIX/.nvitk-post-link.log`. Run the tests with `pytest tests/`
and build the docs with `sphinx-build -b html docs docs/_build/html`.

### GPU stack

CUDA 13. `torch`/`torchvision` from the `cu130` PyTorch wheel index, plus `cupy-cuda13x`,
`cutensor-cu13`, `nvidia-nccl-cu13`, all pip-installed. Requires **Python 3.11** (the recipe
pins 3.11). `install.sh --cpu` swaps in the CPU wheels from the `cpu` index instead and skips
the CUDA packages.

The deep-learning pipelines ({doc}`pipelines/topbrain`, {doc}`pipelines/cvit`) and
`nvitk.nn` need this stack: `torch` plus nnU-Net's own dependencies (`batchgenerators`,
`batchgeneratorsv2`, `blosc2`, `acvl-utils`), all pulled in by the `nnunetv2` requirement. The
vendored nnU-Net / nnssl builds under `src/nvitk/pipes/_engines/` are **not** installed; the
pipelines put them on the training subprocess's `PYTHONPATH`, so run them from a source checkout
or the cluster container. Check an environment with
`python -c "import torch, nnunetv2; print(torch.cuda.is_available())"`.

### VMTK

The centerline, surface and 4D-flow tracing tools ({doc}`api/vmtk`) need VMTK 1.5.2. No ready
package fits a Python 3.11 / NumPy 2 environment:
- conda-forge's py311 VMTK pins `numpy<2`;
- upstream publishes PyPI wheels only for CPython ≥ 3.12.

The post-link step therefore installs a **cp311 wheel of VMTK 1.5.2**, built from the upstream
sdist against the same `vtk==9.6.2` wheel that upstream uses. VTK is pinned to that version for
this reason.

| Step | What happens |
|---|---|
| 1 | Download the wheel from the `vmtk-1.5.2-cp311` GitHub release and verify its sha256 |
| 2 | If that fails: build the wheel from the sdist with `recipe/vmtk/build_wheel.sh native` (needs gcc/g++, git and network; takes a while) |
| 3 | Install it with `--no-deps` and check that `vmtk.vmtkscripts` and `vtkvmtk` import |

To use another wheel, set `NVITK_VMTK_WHEEL_URL` (a URL or a local path) and
`NVITK_VMTK_WHEEL_SHA256`; an empty checksum skips the check. To rebuild the portable release
wheel (manylinux_2_28, needs docker):

```bash
bash recipe/vmtk/build_wheel.sh manylinux dist/vmtk
```

Check an environment with
`python -c "from nvitk.vmtk import vmtk_version; print(vmtk_version())"`.

### FastSurfer

FastSurfer **v2.5.4** (the newest release that runs on Python 3.11) is installed by the post-link
step into `$CONDA_PREFIX/share/nvitk/fastsurfer`: its source tree, its network checkpoints
(≈320 MB, fetched with a timeout-guarded downloader because one of FastSurfer's mirrors stalls),
and its Python dependencies (`monai`, `lapy`, `yacs`, `neuroreg`, `meshpy`, `pyrr`,
`whippersnappy`). A failure there does not fail the install; `nvitk-fastsurfer install` retries.
Set `NVITK_FASTSURFER=0` (conda) or pass `--no-fastsurfer` (`install.sh`) to skip it. Its
surface and longitudinal pipelines also need FreeSurfer 7.4.1 + license (see below).

### FSL

nvitk uses the FSL installation `$FSLDIR` points to. Without one, `nvitk-fsl install` (run by
`install.sh` by default, offered by the GUI the first time an FSL tool finds no FSL) creates
**nvitk's own FSL core** in `$CONDA_PREFIX/share/nvitk/fsl`: FSL's conda packages for the chosen
components (structural, registration, diffusion, functional and `asl` — BASIL / oxford_asl for
ASL perfusion), at exactly the versions of FSL release 6.0.7.23 (a few GB; the GPU builds of eddy,
bedpostx and probtrackx2 with `--cuda`). It is a separate prefix — FSL brings its own Python — and
it needs conda, which is why the conda package's post-link step does not create it.
`nvitk-fsl info` shows which installation is in use; `NVITK_FSLDIR` or `nvitk-fsl use DIR`
overrides `$FSLDIR`. FSL is free for non-commercial use only — see its
[license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html).

### FreeSurfer topology tools

`nvitk-freesurfer` and the GUI's *FreeSurfer* category wrap FreeSurfer's topology programs
(`mri_pretess`, `mri_mc`, `mris_euler_number`, `mris_sphere`, `mris_fix_topology`,
`mris_topo_fixer`, `defect-seg` …) for any mask. They use whichever FreeSurfer is found:
`$FREESURFER_HOME`, then `freesurfer_home` in `external_tools.json`, `/usr/local/freesurfer`,
`/opt/freesurfer`, `~/.local/opt/freesurfer`, `~/freesurfer` or FastSurfer's installer folder. Any
version that has the programs works, 5.3 included. `nvitk-freesurfer info` reports the
installation, its version, its license and the programs present. FreeSurfer's programs need its
(free) license file: `$FS_LICENSE`, `nvitk-freesurfer use --license FILE`, or `license.txt` in the
installation. FreeSurfer 5.x reads only the latter, so nvitk links a license kept elsewhere into
an overlay of the installation under `~/.cache/nvitk`. FastSurfer's surface pipelines keep
looking only for FreeSurfer 7.4.1, as before.

### On the cluster

FastSurfer and FSL jobs (`--submit sge`, the GUI's *Run SGE*) run in the nvitk Singularity
container, whose definition (`registry/containers/projects/nvitk`) installs the FSL components
above (`FSLDIR=/opt/conda`), FastSurfer v2.5.4 with its checkpoints, and FreeSurfer 7.4.1 pruned to
what recon-surf runs (`FREESURFER_HOME=/opt/freesurfer`; the license is bound per job, never baked
in). Rebuild the container and point `paths.nvitk_container` in `sge.json` at it; the nvitk source
the jobs bind (`paths.nvitk_src_dir`) has to be a version with these tools.

## External prerequisites (not installed by conda or `install.sh`)

Separately licensed or with no package-manager distribution — install these yourself and
confirm they're on `PATH`/the relevant environment variable. nvitk's CLIs notify you when a
needed-but-missing external tool is required for the operation you're running.

| Dependency | Version | Used by | Install |
|---|---|---|---|
| **FSL** (full suite) | 6.0.7.x | anything beyond nvitk's FSL core (FEAT GUI, FSLeyes …) | [fsl.fmrib.ox.ac.uk/fsldownloads](https://fsl.fmrib.ox.ac.uk/fsldownloads/) — official `fslinstaller.py`; non-commercial license required. `nvitk-fsl install` covers the tools nvitk wraps. |
| **FreeSurfer** | 7.4.1 | FastSurfer surfaces / longitudinal, Desikan atlas lookups (`$FREESURFER_HOME`) | `nvitk-fastsurfer install-freesurfer`, or [surfer.nmr.mgh.harvard.edu](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) — free registration (license file) required |
| **SPM12** | r7771 | The QVT+ MATLAB reference pipeline | [fil.ion.ucl.ac.uk/spm/software/spm12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) |

```{seealso}
The full CLI catalog is in {doc}`api/cli-catalog`, and every command's `--help` output is
also embedded throughout the {doc}`api/index`, {doc}`gui/index`, and {doc}`pipelines/index`
pages.
```
