# Configuration

nvitk keeps every site-specific path in JSON configuration files, never in the source. A fresh
install therefore knows *what* it needs but not *where* anything is, and the first thing to do
after installing is create a configuration.

```bash
nvitk-config init        # writes ~/.config/nvitk/{sge,settings,xnat}.json from templates
nvitk-config path        # show every directory searched, and which one is in use
nvitk-config validate    # list settings still holding <PLACEHOLDER> values
nvitk-config show        # resolved values, and the file each came from
```

## The three files

| File | Holds |
|---|---|
| `sge.json` | Cluster/SGE settings and every pipeline's data roots (DICOM, NIfTI, results, model weights, containers). |
| `settings.json` | The dataset root (`db.root`) and the atlas directory. |
| `xnat.json` | XNAT server, project and how to obtain credentials. |

## Where they are looked for

The first directory that exists wins — the candidates are **not** merged, so "which file am I
actually using?" always has one answer, and `nvitk-config path` prints it.

| # | Location | Typical use |
|---|---|---|
| 1 | `--config-dir PATH` | One-off run against a different configuration |
| 2 | `$NVITK_SGE_JSON` / `$NVITK_SETTINGS_JSON` / `$NVITK_XNAT_CONFIG` | Override a single file |
| 3 | `$NVITK_CONFIG_DIR` | Per-shell or per-job configuration |
| 4 | `$NVITK_HOME/.nvitk` | Legacy; still honoured |
| 5 | `~/.config/nvitk` | **Default for an installed package** |
| 6 | `~/.nvitk` | Per-user alternative |
| 7 | `./.nvitk` | Per-project, relative to the working directory |
| 8 | `<source checkout>/.nvitk` | Developers working from a clone |

`--config-dir` also exports `$NVITK_CONFIG_DIR`, so SGE jobs, containers and any subprocess
inherit the same configuration as the command that launched them.

```{note}
Prefer `import … as cfg` + `cfg.CONSTANT` over `from … import CONSTANT` in nvitk code.
Configuration constants resolve on attribute access, so the former follows a late
`--config-dir` while the latter snapshots the value at import time.
```

## How a path is resolved

Three independent layers:

1. **Which file** — the table above.
2. **What it says** — `sge.json` is read once and cached.
3. **Which value wins** — per pipeline, as below.

For `--submit local`, a CLI flag beats configuration:

```
--nifti-root  →  pipelines.<id>_paths.local_nifti_root  →  error
```

For `--submit sge`, configuration beats the CLI flag — deliberately, so a cluster run cannot be
pointed at a workstation path by a leftover flag:

```
pipelines.<id>_paths.cluster_nifti_root  →  --nifti-root  →  error
```

```{warning}
That inversion is intentional but surprising: under `--submit sge`, `--nifti-root` is ignored
whenever `cluster_nifti_root` is set. Change the configuration, not the flag.
```

Nothing falls back to a built-in path. An unset-but-required value raises immediately, naming
the key and every location searched:

```
Required setting "pipelines.qvtpy_paths.cluster_nifti_root" is not configured in sge.json.
Looked in: $NVITK_CONFIG_DIR (…); ~/.config/nvitk/sge.json; …
Run `nvitk-config init` to create a starter configuration, then set the key.
```

## Which keys a pipeline needs

Only configure what you run — `nvitk-config init` writes every section, and unused ones can be
deleted.

| Section | Used by |
|---|---|
| `paths` | Everything: cluster source tree, container image, job-script/log/err roots, host aliases |
| `defaults` | Baseline SGE project/account/queue/memory for all pipelines |
| `pipelines.qvtpy`, `pipelines.qvtpy_paths` | {doc}`pipelines/qvtpy` |
| `pipelines.pesa_fat_ct_pet`, `pipelines.pesa_fat_dixon`, `pipelines.pesa_fat_paths` | {doc}`pipelines/pesa-fat` |
| `pipelines.bbtpy_paths` | `nvitk-bbtpy` |
| `pipelines.topbrain`, `pipelines.topbrain_paths` | {doc}`pipelines/topbrain` |
| `pipelines.eicab`, `pipelines.totalsegmentator` | The external segmentation engines |
| `pipelines.image_tools` | The `nvitk-morph` / `-filter` / `-measure` / … CLIs under `--submit sge` |
| `pipelines.voxelwise` | `nvitk-voxelwise` |

`settings.json`'s `db.root` is needed by anything touching the dataset, including
{doc}`the Stats GUI <stats-gui/index>`.

`db.statmodels_root` is where the Stats GUI saves model configurations, reports and exports.
It is deliberately separate from the dataset: saved models are a researcher's own working
output, written far more often than the dataset changes, and are usually better placed on a
backed-up share than inside a DVC-tracked dataset. When unset it falls back to
`<db.root>/nvitk-statmodels`, which is where models lived before the setting existed.

```{note}
This directory is also a trust boundary. A saved config can define derived columns as Python
expressions; the Stats GUI evaluates them without prompting for configs found under
`statmodels_root`, and asks first for configs opened from anywhere else. Pointing the setting
at a shared directory therefore extends that trust to everyone who can write there.
```

## Credentials

Never put a password in `xnat.json`. nvitk resolves credentials in this order, and any of them
is a better home than the config file:

1. `~/.netrc` (set `netrc_file`; `chmod 600`)
2. the system keyring (`"password_keyring": true` — service `nvitk`, key `xnat:<server>`)
3. `$XNAT_USER` / `$XNAT_PASSWORD`

`$XNAT_SERVER` and `$XNAT_PROJECT` also override the file, so a shared profile can be committed
while per-user details stay in the environment.

## For the CNIC team: the private configuration submodule

The real configuration — actual cluster paths, dataset roots, the XNAT server — lives in a
**private repository** mounted at `.nvitk/`, not in this public repo:

```bash
git submodule update --init .nvitk
```

Because `<source checkout>/.nvitk` is search location #8, a clone with the submodule checked
out is configured with no further steps.

```{note}
This repository already carries private submodules (`registry/containers`, `registry/models`),
so `git clone --recurse-submodules` fails for anyone outside CNIC. That is expected: the
submodules are optional, and nvitk works without them. Public users configure nvitk with
`nvitk-config init` instead.
```

### The dataset, via DVC

The dataset is far too large for git, so it is tracked with [DVC](https://dvc.org). DVC keeps a
small text pointer in git and stores the actual bytes on CNIC storage:

```
dataset/nvitk-dataset/tables/    real files, gitignored, never in git history
dataset/nvitk-dataset/tables.dvc ~5 lines: an md5 hash, a size, a path — this IS in git
.dvc/config                      the remote location (shared, committed)
```

Each pointer records a hash of its directory's contents. `dvc push` uploads anything whose hash
is not already on the remote; `dvc pull` downloads whatever the checked-out pointer names.
Because pointers are versioned in git alongside the code, checking out an older commit and
pulling gives the dataset *as it was at that commit* — the property that makes an analysis
reproducible.

The pointers are public and the content is not. A hash reveals nothing, so access is enforced
where the bytes are: only someone with the storage mounted can download them. That is what lets
`nvitk-dataset pull` work from a plain conda install with no repository clone — see
{ref}`get-the-data`.

Three targets are tracked separately so a routine pull stays small:

| Target | Size | Pulled by default |
|---|---|---|
| `catalog` | ~150 KB | yes |
| `tables` | ~19 MB | yes |
| `cache` (SQLite index) | ~1.3 GB | only with `--all` |

The index is derived from the tables and rebuilds in about 15 seconds
(`python -m nvitk.db.sqlite_index --dataset-root <root>`), so transferring it is usually the
slower option. It is tracked anyway, for anyone who would rather download than rebuild.

Maintaining it, from a repository clone:

```bash
dvc pull                                  # fetch what the current commit names
dvc status                                # is the working data in sync with its pointers?
dvc add dataset/nvitk-dataset/tables      # after changing data: re-hash
dvc push                                  # upload, then commit the updated .dvc file
```
