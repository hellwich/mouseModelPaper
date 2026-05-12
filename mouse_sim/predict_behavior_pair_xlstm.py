#!/usr/bin/env python3
"""Predict pairwise mouse behaviors from 3D coordinates using a saved checkpoint.

This script is intended to work with checkpoints produced by train_behavior_pair.py.

Typical usage:

  python predict_behavior_pair.py \
    --ckpt out_pair_train/behavior_pair_lstm.pt \
    --coords-3d out_pair/coords_3d.csv \
    --out-dir out_pair_pred
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

try:
    import train_behavior_pair as tbp
except ImportError:
    import train_behavior_pair_xlstm as tbp


def _csv_has_column(path: str, col: str) -> bool:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, [])
    return col in header


def _build_model_from_ckpt(ckpt: Dict[str, object]) -> torch.nn.Module:
    model_type = str(ckpt.get("model_type"))
    in_dim = int(ckpt.get("in_dim"))
    cfg = ckpt.get("model_cfg") or {}

    if model_type == "lstm":
        backbone = tbp.LSTMBackbone(
            in_dim=in_dim,
            hidden=int(cfg.get("hidden", 128)),
            layers=int(cfg.get("layers", 2)),
            bidir=bool(cfg.get("bidir", True)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    elif model_type == "transformer":
        backbone = tbp.TransformerBackbone(
            in_dim=in_dim,
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 4)),
            layers=int(cfg.get("layers", 4)),
            ff=int(cfg.get("ff", 256)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    elif model_type == "xlstm":
        backbone = tbp.xLSTMBackbone(
            in_dim=in_dim,
            d_model=int(cfg.get("d_model", 128)),
            num_blocks=int(cfg.get("num_blocks", 4)),
            context_length=int(cfg.get("context_length", 256)),
            backend=str(cfg.get("backend", "cuda")),
            num_heads=int(cfg.get("nhead", 4)),
            conv1d_kernel_size=int(cfg.get("conv1d_kernel_size", 4)),
        )
    else:
        raise ValueError(
            f"Unsupported model_type in checkpoint: {model_type!r}. "
            "Expected one of: lstm, transformer, xlstm."
        )

    model = tbp.MultiTaskHead(backbone, out_dim=backbone.out_dim)
    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing 'model_state' dict")
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser("Predict pair behaviors (2 mice) from 3D coords using a saved checkpoint")

    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt produced by train_behavior_pair.py")
    ap.add_argument(
        "--coords-3d",
        nargs="+",
        required=True,
        help="One or more coords_3d.csv paths (same format as training).",
    )
    ap.add_argument("--out-dir", default="predict_behavior_pair_out", help="Directory to write predicted ethograms")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")

    ap.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override window size (default: from checkpoint)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride (default: from checkpoint)",
    )
    ap.add_argument(
        "--smooth-k",
        type=int,
        default=None,
        help="Override interaction smoothing window (odd). Default: from checkpoint.",
    )

    ap.add_argument("--no-ethograms", action="store_true", help="Do not write ethogram CSVs")
    ap.add_argument("--report", action="store_true", help="If labels exist, print classification reports")
    ap.add_argument(
        "--no-labels",
        action="store_true",
        help="Treat input as unlabeled (behavior column may be absent); metrics won't be computed.",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint file did not contain a dict")

    model = _build_model_from_ckpt(ckpt)
    model.to(device)
    model.eval()

    node_order = ckpt.get("node_order")
    if not isinstance(node_order, list) or not node_order:
        raise ValueError("Checkpoint missing 'node_order'")

    scale_mm = float(ckpt.get("scale_mm", 1.0))
    tail_root = str(ckpt.get("tail_root", "tail_root"))
    head = str(ckpt.get("head", "head"))
    nose = str(ckpt.get("nose", "nose_tip"))
    use_vel = bool(ckpt.get("use_vel", True))

    window = int(args.window if args.window is not None else ckpt.get("window", 20))
    stride = int(args.stride if args.stride is not None else ckpt.get("stride", 1))
    smooth_k = int(args.smooth_k if args.smooth_k is not None else ckpt.get("smooth_k", 1))
    if smooth_k > 1 and smooth_k % 2 == 0:
        raise ValueError("--smooth-k must be odd")

    export_ethograms = (not args.no_ethograms)

    for p in args.coords_3d:
        has_behavior = (not args.no_labels) and _csv_has_column(p, "behavior")

        run_raw = tbp.load_coords_3d_csv_pair(
            p,
            node_order=list(node_order),
            scale_mm=scale_mm,
            require_behavior=has_behavior,
        )
        run = tbp.preprocess_run(run_raw, tail_root=tail_root, head=head, nose=nose, use_vel=use_vel)

        p_int, p0, p1 = tbp.predict_probs_per_run(model, run, window=window, stride=stride, device=device)
        int_pred = (p_int >= 0.5).astype(np.int64)
        if smooth_k > 1:
            int_pred = tbp._majority_smooth_binary(int_pred, smooth_k)
        rear0_pred = (p0 >= 0.5).astype(np.int64)
        rear1_pred = (p1 >= 0.5).astype(np.int64)

        if export_ethograms:
            if has_behavior:
                tbp.export_ethogram_ground_truth(args.out_dir, run_raw)
            out_path = tbp.export_ethogram_prediction(
                args.out_dir,
                run_raw,
                int_prob=p_int,
                int_pred=int_pred,
                rear0_prob=p0,
                rear0_pred=rear0_pred,
                rear1_prob=p1,
                rear1_pred=rear1_pred,
            )
            print(f"wrote: {out_path}")

        if args.report and has_behavior:
            y_int, y0, y1, mask = tbp.derive_multitask_labels(run_raw.beh)
            maskb = mask.astype(bool)
            from sklearn.metrics import classification_report, f1_score

            print("\n==", run_raw.name, "==")
            print(f"Interaction f1: {f1_score(y_int, int_pred, pos_label=1):.4f}")
            print(classification_report(y_int, int_pred, digits=4))
            if maskb.any():
                print(f"Rear0 f1 (non-interaction): {f1_score(y0[maskb], rear0_pred[maskb], pos_label=1):.4f}")
                print(classification_report(y0[maskb], rear0_pred[maskb], digits=4))
                print(f"Rear1 f1 (non-interaction): {f1_score(y1[maskb], rear1_pred[maskb], pos_label=1):.4f}")
                print(classification_report(y1[maskb], rear1_pred[maskb], digits=4))

    if export_ethograms:
        print(f"Ethograms written to: {args.out_dir}")


if __name__ == "__main__":
    main()
