"""
Attention-usage probe worker — runs **inside** the in-tree nnU-Net subprocess.

Description
-----------
Imported only by ``python -m nvitk.pipes.cvit.probe_worker`` launched by
:mod:`nvitk.pipes.cvit.stage3b_probe` with :func:`nvitk.pipes._engines.env.nnunet_env`, so
``nnunetv2`` here is the in-tree build that knows the CViT trainers.

For each fold, the cases that fold **held out** (``splits_final.json``) are:

1. read with the plan's own reader/writer (nnU-Net axis order) and preprocessed **once**;
2. predicted with the trained fold under every requested intervention
   (:func:`nvitk.nn.cvit.probe.intervene`) using the same sliding-window / Gaussian / mirroring
   settings as inference, resampled back to the original grid;
3. scored against the reference label with
   :func:`nvitk.measure.segmentation_metrics.evaluate_case` (spacing from the reader, mm).

The first held-out case of each fold also yields passive attention statistics
(:func:`nvitk.nn.cvit.probe.attention_stats`) on a centre crop of its preprocessed volume at the
training patch size, with the plan's target spacing.

Outputs (``--out-dir``)
-----------------------
``rows.jsonl``   one JSON object per case × intervention × label
``stats.json``   per-fold passive statistics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Interventions flip module attributes; a compiled graph could have specialised on them.
os.environ["nnUNet_compile"] = "f"

import numpy as np  # noqa: E402  (host-only worker: nnU-Net hands us NumPy arrays)
import torch  # noqa: E402

from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy  # noqa: E402
from nnunetv2.inference.export_prediction import (  # noqa: E402
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

from nvitk.measure.segmentation_metrics import evaluate_case  # noqa: E402
from nvitk.nn.cvit.probe import attention_stats, intervene  # noqa: E402
from nvitk.nn.cvit.weights import _unwrap  # noqa: E402


def parse_interventions(names: list[str], radii: list[float], depth: int, layer_sweep: bool,
                        has_skips: bool) -> list[tuple[str, str, dict]]:
    """``(label, mode, kwargs)`` for every intervention to run."""
    out: list[tuple[str, str, dict]] = []
    for name in names:
        if name == "attn_local":
            out += [(f"attn_local@{r:g}mm", "attn_local", {"radius_mm": r}) for r in radii]
        elif name == "skips_off" and not has_skips:
            continue
        else:
            out.append((name, name, {}))
    if layer_sweep:
        out += [(f"attn_off@L{i:02d}", "attn_off", {"layers": [i]}) for i in range(depth)]
    if not any(label == "full" for label, _, _ in out):
        out.insert(0, ("full", "full", {}))
    return out


def centre_crop(data: torch.Tensor, patch: list[int]) -> torch.Tensor:
    """``(C, *S)`` → ``(1, C, *patch)`` centre crop, zero-padded where the volume is smaller."""
    c, *shape = data.shape
    out = torch.zeros((1, c, *patch), dtype=data.dtype)
    src, dst = [], []
    for s, p in zip(shape, patch):
        if s >= p:
            a = (s - p) // 2
            src.append(slice(a, a + p)); dst.append(slice(0, p))
        else:
            a = (p - s) // 2
            src.append(slice(0, s)); dst.append(slice(a, a + s))
    out[(0, slice(None), *dst)] = data[(slice(None), *src)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--interventions", default="full,attn_off,attn_local,attn_uniform,transformer_off,tokens_off,skips_off")
    ap.add_argument("--local-radii-mm", default="10,30")
    ap.add_argument("--layer-sweep", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0, help="Per fold; 0 = all held-out cases.")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--n-queries", type=int, default=256)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_json = json.loads((raw / "dataset.json").read_text())
    labels = sorted(int(v) for k, v in dataset_json["labels"].items() if int(v) != 0 and k != "ignore")
    n_channels = len(dataset_json["channel_names"])
    splits = json.loads(Path(args.splits).read_text())
    radii = [float(r) for r in args.local_radii_mm.split(",") if r.strip()]
    names = [n.strip() for n in args.interventions.split(",") if n.strip()]
    device = torch.device(args.device)

    rows_path = out / "rows.jsonl"
    rows_path.write_text("")
    stats_all: dict[str, dict] = {}

    for fold in [f.strip() for f in args.folds.split(",") if f.strip()]:
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not args.no_mirror,
                                    perform_everything_on_device=device.type == "cuda", device=device,
                                    verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(args.run_dir, use_folds=(int(fold) if fold.isdigit() else fold,),
                                                       checkpoint_name=args.checkpoint)
        # nnU-Net only moves the network inside its sliding window; the attention statistics run
        # before any prediction, so put it on the device here.
        predictor.network = predictor.network.to(device).eval()
        net = _unwrap(predictor.network)
        has_skips = any(getattr(net.decoder, "enabled", ()))
        plan_list = parse_interventions(names, radii, len(net.encoder.blocks), args.layer_sweep, has_skips)
        cm, pm = predictor.configuration_manager, predictor.plans_manager
        rw = pm.image_reader_writer_class()
        cases = splits[int(fold)]["val"] if fold.isdigit() else []
        if args.max_cases:
            cases = cases[: args.max_cases]
        print(f"[probe] fold {fold}: {len(cases)} case(s) x {len(plan_list)} intervention(s)", flush=True)

        for k, case in enumerate(cases):
            files = [str(raw / "imagesTr" / f"{case}_{c:04d}{dataset_json['file_ending']}") for c in range(n_channels)]
            img, props = rw.read_images(files)
            ref, _ = rw.read_seg(str(raw / "labelsTr" / f"{case}{dataset_json['file_ending']}"))
            ref = ref[0].astype(np.int32)
            ppa = PreprocessAdapterFromNpy([img], [None], [props], [None], pm, predictor.dataset_json, cm,
                                           num_threads_in_multithreaded=1, verbose=False)
            dct = next(ppa)
            data = dct["data"]

            if k == 0:
                x = centre_crop(data, list(cm.patch_size)).to(device)
                stats = attention_stats(net, x, spacing_mm=cm.spacing, n_queries=args.n_queries)
                stats["case"] = case
                stats_all[str(fold)] = stats

            for label, mode, kw in plan_list:
                if mode == "attn_local":
                    kw = {**kw, "spacing_mm": cm.spacing}
                with intervene(predictor.network, mode, **kw):
                    logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
                seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits, pm, cm, predictor.label_manager, dct["data_properties"])
                m = evaluate_case(ref, seg.astype(np.int32), case_id=case, labels=labels,
                                  spacing=props["spacing"])
                with open(rows_path, "a", encoding="utf-8") as fh:
                    base = {"case": case, "fold": str(fold), "intervention": label, "mode": mode}
                    fh.write(json.dumps({**base, "label": "all", **{
                        k2.replace("class_avg_", ""): v for k2, v in m.aggregate.items()}}) + "\n")
                    for value, entry in m.per_class.items():
                        fh.write(json.dumps({**base, "label": int(value), **entry}) + "\n")
            print(f"[probe] fold {fold} case {k + 1}/{len(cases)} {case} done", flush=True)

    (out / "stats.json").write_text(json.dumps(stats_all, indent=2) + "\n")


if __name__ == "__main__":
    main()
