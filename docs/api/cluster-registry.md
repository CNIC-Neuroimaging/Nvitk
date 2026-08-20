# Cluster & Registry

Two small supporting modules used by every pipeline's `--submit sge` path and by the
Singularity-based segmentation engines.

## Cluster (SGE)

`nvitk.cluster` — submission, chunking, config resolution, and remote (SSH) execution for
Sun Grid Engine:

| Module | Purpose |
|---|---|
| `sge` | Core job submission (`qsub`) wrapper. |
| `sge_chunk` | Splits large subject batches into array-job chunks under a per-user job cap. |
| `sge_json` | Resolves `.nvitk/sge.json` (ascends from cwd to the repo root, or `NVITK_HOME`/`NVITK_SGE_JSON`) and merges it over each pipeline's Python defaults. |
| `sge_remote`, `remote_submit`, `remote_transfer` | SSH-based remote submission and file transfer for clusters not reachable from the local `qsub`. |

Every pipeline's `--submit local|sge` flag and every module CLI's `--submit` option go
through this layer — see {doc}`../pipelines/index` for the concrete per-subject dispatch
pattern.

## Registry

`nvitk.registry` — Singularity container and model registry helpers (`containers`,
`cli_sync_sge`) backing the segmentation engines that run inside containers (eICAB) rather
than as native Python imports.

```{seealso}
Full generated reference: [`nvitk.cluster`](../autoapi/nvitk/cluster/index),
[`nvitk.registry`](../autoapi/nvitk/registry/index).
```
