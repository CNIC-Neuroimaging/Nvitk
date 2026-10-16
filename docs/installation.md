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

## External prerequisites (not installed by conda or `install.sh`)

Separately licensed or with no package-manager distribution — install these yourself and
confirm they're on `PATH`/the relevant environment variable. nvitk's CLIs notify you when a
needed-but-missing external tool is required for the operation you're running.

| Dependency | Version | Used by | Install |
|---|---|---|---|
| **FSL** | 6.0.7.19 | `nvitk-flirt`, registration CLIs, the {doc}`QVTPy pipeline <pipelines/qvtpy>` | [fsl.fmrib.ox.ac.uk/fsldownloads](https://fsl.fmrib.ox.ac.uk/fsldownloads/) — official `fslinstaller.py`; non-commercial license required |
| **FreeSurfer** | 7.x | Desikan atlas lookups (`$FREESURFER_HOME`) | [surfer.nmr.mgh.harvard.edu](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) — free registration required |
| **SPM12** | r7771 | The QVT+ MATLAB reference pipeline | [fil.ion.ucl.ac.uk/spm/software/spm12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) |

```{seealso}
The full CLI catalog is in {doc}`api/cli-catalog`, and every command's `--help` output is
also embedded throughout the {doc}`api/index`, {doc}`gui/index`, and {doc}`pipelines/index`
pages.
```
