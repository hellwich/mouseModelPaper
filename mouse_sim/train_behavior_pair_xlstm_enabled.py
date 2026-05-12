#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Classical baselines
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report


# ----------------------------
# Helpers
# ----------------------------

def nlabel(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no cuda


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt((v * v).sum()) + eps)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class PairRun:
    name: str
    times: np.ndarray         # (T,)
    pos: np.ndarray           # (T,2,K,3) world coords
    beh: np.ndarray           # (T,2) in {0,1,2}
    node_order: List[str]     # (K,)
    mouse_ids: Tuple[int, int]


def load_coords_3d_csv_pair(
    path: str,
    node_order: Optional[List[str]] = None,
    mouse_ids: Optional[Tuple[int, int]] = None,
    scale_mm: float = 1.0,
    require_behavior: bool = True,
) -> PairRun:
    """
    Expects coords_3d.csv with columns:
      frame,time,mouse_id,behavior,node,x,y,z
    Backward compatible: if mouse_id is missing, assumes single mouse_id=0 (will error later if you need 2).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        have = set(r.fieldnames or [])
        required = {"frame", "time", "node", "x", "y", "z"}
        if require_behavior:
            required = required | {"behavior"}
        if not required.issubset(have):
            raise ValueError(f"{path} missing required columns. Have: {r.fieldnames}")
        has_mouse_id = "mouse_id" in have
        has_behavior = "behavior" in have

        rows = [row for row in r]

    if not rows:
        raise ValueError(f"{path}: no rows")

    frames = sorted(set(int(rr["frame"]) for rr in rows))
    if node_order is None:
        node_order = sorted(set(nlabel(rr["node"]) for rr in rows))
    node_to_i = {n: i for i, n in enumerate(node_order)}

    if has_mouse_id:
        mids_all = sorted(set(int(rr["mouse_id"]) for rr in rows))
    else:
        mids_all = [0]

    if mouse_ids is None:
        if len(mids_all) < 2:
            raise ValueError(
                f"{path}: expected 2 mice (mouse_id column with at least two ids), found {mids_all}."
            )
        mouse_ids = (mids_all[0], mids_all[1])

    mid_to_m = {mouse_ids[0]: 0, mouse_ids[1]: 1}

    # Group by (frame, mouse_id)
    by_fm: Dict[Tuple[int, int], List[dict]] = {}
    for rr in rows:
        fidx = int(rr["frame"])
        mid = int(rr["mouse_id"]) if has_mouse_id else 0
        if mid not in mid_to_m:
            continue  # ignore other mice ids
        by_fm.setdefault((fidx, mid), []).append(rr)

    T = len(frames)
    K = len(node_order)
    times = np.zeros((T,), dtype=np.float32)
    pos = np.zeros((T, 2, K, 3), dtype=np.float32)
    beh = np.zeros((T, 2), dtype=np.int64)

    # Fill
    for ti, fidx in enumerate(frames):
        # time: pick from any present row
        # Prefer mouse_ids[0] if present, else first available
        key0 = (fidx, mouse_ids[0])
        key1 = (fidx, mouse_ids[1])
        fr0 = by_fm.get(key0)
        fr1 = by_fm.get(key1)
        fr_any = fr0 or fr1
        if not fr_any:
            raise ValueError(f"{path}: frame {fidx} has no data for selected mouse_ids={mouse_ids}")
        times[ti] = float(fr_any[0]["time"])

        for mid, m in mid_to_m.items():
            fr = by_fm.get((fidx, mid))
            if not fr:
                raise ValueError(f"{path}: missing data for frame={fidx} mouse_id={mid}")
            beh[ti, m] = int(fr[0].get("behavior", 0))
            for rr in fr:
                node = nlabel(rr["node"])
                if node not in node_to_i:
                    continue
                j = node_to_i[node]
                pos[ti, m, j, 0] = float(rr["x"]) * scale_mm
                pos[ti, m, j, 1] = float(rr["y"]) * scale_mm
                pos[ti, m, j, 2] = float(rr["z"]) * scale_mm

    return PairRun(
        name=os.path.basename(os.path.dirname(path)) or os.path.basename(path),
        times=times,
        pos=pos,
        beh=beh,
        node_order=node_order,
        mouse_ids=mouse_ids,
    )


# ----------------------------
# Pair-frame features
# ----------------------------

def pair_frame_transform(
    tail0: np.ndarray, tail1: np.ndarray, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pair frame from tail_root positions (world coords).
    Origin: midpoint.
    x-axis: tail0->tail1 projected onto xy.
    z-axis: world up.
    Returns (R, origin) where R is 3x3 with columns [xhat, yhat, zhat].
    """
    o = 0.5 * (tail0 + tail1)
    dx = tail1[0] - tail0[0]
    dy = tail1[1] - tail0[1]
    v = np.array([dx, dy, 0.0], dtype=np.float64)
    n = safe_norm(v, eps)
    if n < 1e-6:
        xhat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        xhat = v / n
    zhat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yhat = np.cross(zhat, xhat)
    yn = safe_norm(yhat, eps)
    if yn < 1e-6:
        yhat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        yhat = yhat / yn
    # Re-orthogonalize xhat to be safe
    xhat = np.cross(yhat, zhat)
    R = np.stack([xhat, yhat, zhat], axis=1)
    return R, o


def to_pair_frame(pos_world: np.ndarray, node_order: List[str], tail_root: str) -> np.ndarray:
    """
    pos_world: (T,2,K,3)
    returns pos_pair: (T,2,K,3) in pair frame per timestep.
    """
    tail_root = nlabel(tail_root)
    idx = {n: i for i, n in enumerate(node_order)}
    if tail_root not in idx:
        raise ValueError(f"tail_root node '{tail_root}' not found in node_order")
    tr = idx[tail_root]

    T, M, K, _ = pos_world.shape
    out = np.zeros_like(pos_world, dtype=np.float32)

    for t in range(T):
        tail0 = pos_world[t, 0, tr].astype(np.float64)
        tail1 = pos_world[t, 1, tr].astype(np.float64)
        R, o = pair_frame_transform(tail0, tail1)
        Rt = R.T
        # transform each mouse/keypoint
        pw = pos_world[t].astype(np.float64)  # (2,K,3)
        pp = (pw - o[None, None, :]) @ Rt.T    # (2,K,3)
        out[t] = pp.astype(np.float32)
    return out


def make_pair_features(
    pos_pair: np.ndarray,
    node_order: List[str],
    nose: str,
    head: str,
    tail_root: str,
    use_vel: bool = True,
) -> np.ndarray:
    """
    pos_pair: (T,2,K,3)
    Returns X: (T,F) with F = 2*K*3 (+ vel same size) + extras.
    Extras: [nose_dist, tail_dist] in pair frame (scalars).
    """
    nose = nlabel(nose)
    head = nlabel(head)
    tail_root = nlabel(tail_root)
    idx = {n: i for i, n in enumerate(node_order)}
    for need in (nose, head, tail_root):
        if need not in idx:
            raise ValueError(f"required node '{need}' not found in node_order")

    T, M, K, _ = pos_pair.shape
    p = pos_pair.reshape(T, M * K * 3)

    # Scalars (pair frame)
    nose0 = pos_pair[:, 0, idx[nose], :]
    nose1 = pos_pair[:, 1, idx[nose], :]
    tail0 = pos_pair[:, 0, idx[tail_root], :]
    tail1 = pos_pair[:, 1, idx[tail_root], :]
    nose_dist = np.linalg.norm(nose1 - nose0, axis=1).reshape(T, 1).astype(np.float32)
    tail_dist = np.linalg.norm(tail1 - tail0, axis=1).reshape(T, 1).astype(np.float32)
    extras = np.concatenate([nose_dist, tail_dist], axis=1)  # (T,2)

    if not use_vel:
        return np.concatenate([p, extras], axis=1)

    v = np.zeros_like(p)
    v[1:] = p[1:] - p[:-1]
    return np.concatenate([p, v, extras], axis=1)


def derive_multitask_labels(beh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    beh: (T,2) in {0 run, 1 rear, 2 interact}
    Returns:
      y_int: (T,) 0/1
      y0: (T,) 0/1 (rear vs run), undefined during interaction
      y1: (T,) 0/1
      mask: (T,) 0/1 where 1 means non-interaction frames (use y0/y1)
    """
    y_int = ((beh[:, 0] == 2) | (beh[:, 1] == 2)).astype(np.int64)
    mask = (y_int == 0).astype(np.int64)
    y0 = (beh[:, 0] == 1).astype(np.int64)
    y1 = (beh[:, 1] == 1).astype(np.int64)
    return y_int, y0, y1, mask


# ----------------------------
# Windowing dataset with swap augmentation
# ----------------------------

@dataclass
class PreprocRun:
    name: str
    X_base: np.ndarray     # (T,F)
    X_swap: np.ndarray     # (T,F)
    y_int: np.ndarray      # (T,)
    y0_base: np.ndarray    # (T,)
    y1_base: np.ndarray    # (T,)
    y0_swap: np.ndarray    # (T,)
    y1_swap: np.ndarray    # (T,)
    mask: np.ndarray       # (T,)


class PairWindowDataset(Dataset):
    def __init__(
        self,
        runs: Sequence[PreprocRun],
        window: int,
        stride: int,
        swap_prob: float,
        seed: int,
    ):
        self.runs = list(runs)
        self.window = int(window)
        self.stride = int(stride)
        self.swap_prob = float(swap_prob)
        self.seed = int(seed)
        self.epoch = 0

        self.index: List[Tuple[int, int]] = []  # (run_idx, start)
        for ri, run in enumerate(self.runs):
            T = run.X_base.shape[0]
            if T < self.window:
                continue
            for s in range(0, T - self.window + 1, self.stride):
                self.index.append((ri, s))

        if not self.index:
            raise ValueError("No window samples created. Increase data or reduce window.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def _det_swap(self, i: int) -> bool:
        # Deterministic pseudo-random swap for reproducibility, varied by epoch.
        # Produces a uniform-ish bit from a simple hash.
        x = (i + 1) * 1000003 + (self.seed + 17) * 9176 + (self.epoch + 11) * 2654435761
        # xorshift-ish
        x ^= (x >> 13)
        x ^= (x << 17) & 0xFFFFFFFF
        x ^= (x >> 5)
        # map to [0,1)
        u = (x & 0xFFFFFFFF) / 2**32
        return u < self.swap_prob

    def __getitem__(self, i: int):
        ri, s = self.index[i]
        run = self.runs[ri]
        do_swap = self._det_swap(i)

        if do_swap:
            X = run.X_swap[s:s + self.window]
            y0 = run.y0_swap[s:s + self.window]
            y1 = run.y1_swap[s:s + self.window]
        else:
            X = run.X_base[s:s + self.window]
            y0 = run.y0_base[s:s + self.window]
            y1 = run.y1_base[s:s + self.window]

        y_int = run.y_int[s:s + self.window]
        mask = run.mask[s:s + self.window]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_int, dtype=torch.float32),
            torch.tensor(y0, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


# ----------------------------
# Models: backbones + multitask heads
# ----------------------------

class LSTMBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2,
                 bidir: bool = True, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidir,
        )
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x):  # (B,T,F)
        z, _ = self.lstm(x)
        return z  # (B,T,out_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBackbone(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 4,
                 layers: int = 4, ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_dim = d_model

    def forward(self, x):
        z = self.embed(x)
        z = self.pe(z)
        z = self.enc(z)
        return z


class xLSTMBackbone(nn.Module):
    """
    Requires: pip install xlstm
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        num_blocks: int = 4,
        context_length: int = 256,
        backend: str = "native",
        num_heads: int = 4,
        conv1d_kernel_size: int = 4,
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)

        try:
            from xlstm import (
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                FeedForwardConfig,
            )
        except Exception as e:
            raise RuntimeError(
                "xLSTM selected but 'xlstm' package is not installed.\n"
                "Install:\n  pip install xlstm\n"
                "or editable:\n  git clone https://github.com/NX-AI/xlstm.git && cd xlstm && pip install -e .\n"
            ) from e

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,  # "native" or "cuda"
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=[1],
        )

        self.core = xLSTMBlockStack(cfg)
        self.out_dim = d_model

    def forward(self, x):
        z = self.embed(x)
        z = self.core(z)
        return z


class MultiTaskHead(nn.Module):
    def __init__(self, backbone: nn.Module, out_dim: int):
        super().__init__()
        self.backbone = backbone
        self.h_int = nn.Linear(out_dim, 1)
        self.h0 = nn.Linear(out_dim, 1)
        self.h1 = nn.Linear(out_dim, 1)

    def forward(self, x):
        z = self.backbone(x)  # (B,T,H)
        return self.h_int(z), self.h0(z), self.h1(z)


# ----------------------------
# Baselines
# ----------------------------

def geometric_rule_interaction(
    run: PairRun,
    head: str,
    nose: str,
    tail_root: str,
    scale_mm: float,
    dist_thr: float,
    facing_thr: float,
) -> np.ndarray:
    """
    Simple rule baseline for interaction: nose distance + both facing each other.
    Returns yhat_int (T,) in {0,1}
    """
    head = nlabel(head); nose = nlabel(nose); tail_root = nlabel(tail_root)
    idx = {n: i for i, n in enumerate(run.node_order)}
    for need in (head, nose, tail_root):
        if need not in idx:
            raise ValueError(f"required node '{need}' not found")

    h0 = run.pos[:, 0, idx[head], :].astype(np.float64)
    h1 = run.pos[:, 1, idx[head], :].astype(np.float64)
    n0 = run.pos[:, 0, idx[nose], :].astype(np.float64)
    n1 = run.pos[:, 1, idx[nose], :].astype(np.float64)
    t0 = run.pos[:, 0, idx[tail_root], :].astype(np.float64)
    t1 = run.pos[:, 1, idx[tail_root], :].astype(np.float64)

    f0 = h0 - t0
    f1 = h1 - t1
    # normalize in xy plane for stability
    f0[:, 2] = 0; f1[:, 2] = 0
    f0n = np.linalg.norm(f0, axis=1) + 1e-9
    f1n = np.linalg.norm(f1, axis=1) + 1e-9
    f0 = f0 / f0n[:, None]
    f1 = f1 / f1n[:, None]

    g0 = n1 - n0
    g1 = n0 - n1
    g0[:, 2] = 0; g1[:, 2] = 0
    g0n = np.linalg.norm(g0, axis=1) + 1e-9
    g1n = np.linalg.norm(g1, axis=1) + 1e-9
    g0 = g0 / g0n[:, None]
    g1 = g1 / g1n[:, None]

    d = np.linalg.norm(n1 - n0, axis=1)

    face0 = (f0 * g0).sum(axis=1)
    face1 = (f1 * g1).sum(axis=1)

    yhat = ((d < dist_thr) & (face0 > facing_thr) & (face1 > facing_thr)).astype(np.int64)
    return yhat


def train_eval_logreg_multitask(
    train: Sequence[PreprocRun],
    test: Sequence[PreprocRun],
    C: float,
    smooth_k: int,
) -> None:
    """
    Train 3 separate logistic regressions:
      - interaction on all frames
      - rear0 on non-interaction frames
      - rear1 on non-interaction frames
    """
    def majority_smooth(y: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return y.copy()
        if k % 2 == 0:
            raise ValueError("smooth-k must be odd")
        r = k // 2
        out = y.copy()
        for i in range(len(y)):
            a = max(0, i - r)
            b = min(len(y), i + r + 1)
            out[i] = 1 if (y[a:b].sum() > (b - a) / 2) else 0
        return out

    # Stack per-frame across runs (avoid leakage by splitting by run already)
    Xtr = np.concatenate([r.X_base for r in train], axis=0)
    Xte = np.concatenate([r.X_base for r in test], axis=0)

    yint_tr = np.concatenate([r.y_int for r in train], axis=0)
    yint_te = np.concatenate([r.y_int for r in test], axis=0)

    mask_tr = np.concatenate([r.mask for r in train], axis=0).astype(bool)
    mask_te = np.concatenate([r.mask for r in test], axis=0).astype(bool)

    y0_tr = np.concatenate([r.y0_base for r in train], axis=0)[mask_tr]
    y1_tr = np.concatenate([r.y1_base for r in train], axis=0)[mask_tr]
    y0_te = np.concatenate([r.y0_base for r in test], axis=0)[mask_te]
    y1_te = np.concatenate([r.y1_base for r in test], axis=0)[mask_te]

    Xtr_mask = Xtr[mask_tr]
    Xte_mask = Xte[mask_te]

    def fit_lr(ytr):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, max_iter=1000, class_weight="balanced", solver="lbfgs"
            )),
        ]).fit(Xtr, ytr)

    # Interaction classifier uses all frames
    clf_int = fit_lr(yint_tr)
    yhat_int = clf_int.predict(Xte)

    # Rear classifiers use masked frames (train/test)
    clf0 = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced", solver="lbfgs"
        )),
    ]).fit(Xtr_mask, y0_tr)
    clf1 = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced", solver="lbfgs"
        )),
    ]).fit(Xtr_mask, y1_tr)

    yhat0 = clf0.predict(Xte_mask)
    yhat1 = clf1.predict(Xte_mask)

    if smooth_k > 1:
        # Apply smoothing per run for interaction only (common for ethograms)
        yhat_sm = []
        start = 0
        for r in test:
            T = r.y_int.shape[0]
            yhat_sm.append(majority_smooth(yhat_int[start:start + T], smooth_k))
            start += T
        yhat_int = np.concatenate(yhat_sm, axis=0)

    f1_int = f1_score(yint_te, yhat_int, pos_label=1, zero_division=0)
    f1_0 = f1_score(y0_te, yhat0, pos_label=1, zero_division=0)
    f1_1 = f1_score(y1_te, yhat1, pos_label=1, zero_division=0)
    print(f"[baseline_logreg] f1(interact)={f1_int:.4f}  f1(rear0)={f1_0:.4f}  f1(rear1)={f1_1:.4f}")
    print("Interaction report:")
    print(classification_report(yint_te, yhat_int, digits=4, zero_division=0))
    print("Rear0 report (non-interaction frames):")
    print(classification_report(y0_te, yhat0, digits=4, zero_division=0))
    print("Rear1 report (non-interaction frames):")
    print(classification_report(y1_te, yhat1, digits=4, zero_division=0))


# ----------------------------
# Deep: training + evaluation
# ----------------------------

@torch.no_grad()
def predict_full_runs(
    model: nn.Module,
    runs: Sequence[PreprocRun],
    window: int,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window overlap average logits for each head.
    Returns concatenated arrays:
      y_int_true, y_int_pred, y0_true_masked, y0_pred_masked, y1_true_masked, y1_pred_masked
    """
    model.eval()

    yint_true_all, yint_pred_all = [], []
    y0_true_all, y0_pred_all = [], []
    y1_true_all, y1_pred_all = [], []
    mask_all = []

    for run in runs:
        X = run.X_base
        T = X.shape[0]

        logits_int = np.zeros((T,), dtype=np.float64)
        logits0 = np.zeros((T,), dtype=np.float64)
        logits1 = np.zeros((T,), dtype=np.float64)
        counts = np.zeros((T,), dtype=np.float64)

        def accumulate(s: int, e: int, li, l0, l1):
            logits_int[s:e] += li
            logits0[s:e] += l0
            logits1[s:e] += l1
            counts[s:e] += 1.0

        if T < window:
            xb = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
            li, l0, l1 = model(xb)
            li = li.squeeze(-1).detach().cpu().numpy()[0]
            l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
            l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
            accumulate(0, T, li, l0, l1)
        else:
            for s in range(0, T - window + 1, stride):
                xb = torch.tensor(X[s:s + window][None, :, :], dtype=torch.float32, device=device)
                li, l0, l1 = model(xb)
                li = li.squeeze(-1).detach().cpu().numpy()[0]  # (W,)
                l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
                l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
                accumulate(s, s + window, li, l0, l1)

            uncovered = counts < 0.5
            if uncovered.any():
                xb = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
                li, l0, l1 = model(xb)
                li = li.squeeze(-1).detach().cpu().numpy()[0]
                l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
                l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
                logits_int[uncovered] = li[uncovered]
                logits0[uncovered] = l0[uncovered]
                logits1[uncovered] = l1[uncovered]
                counts[uncovered] = 1.0

        logits_int /= counts
        logits0 /= counts
        logits1 /= counts

        yint_prob = 1.0 / (1.0 + np.exp(-logits_int))
        y0_prob = 1.0 / (1.0 + np.exp(-logits0))
        y1_prob = 1.0 / (1.0 + np.exp(-logits1))

        yint_pred = (yint_prob >= 0.5).astype(np.int64)
        # individual preds are evaluated only on mask frames
        y0_pred = (y0_prob >= 0.5).astype(np.int64)
        y1_pred = (y1_prob >= 0.5).astype(np.int64)

        yint_true_all.append(run.y_int.astype(np.int64))
        yint_pred_all.append(yint_pred)

        mask = run.mask.astype(bool)
        mask_all.append(mask.astype(np.int64))

        y0_true_all.append(run.y0_base[mask].astype(np.int64))
        y0_pred_all.append(y0_pred[mask].astype(np.int64))
        y1_true_all.append(run.y1_base[mask].astype(np.int64))
        y1_pred_all.append(y1_pred[mask].astype(np.int64))

    return (
        np.concatenate(yint_true_all), np.concatenate(yint_pred_all),
        np.concatenate(y0_true_all), np.concatenate(y0_pred_all),
        np.concatenate(y1_true_all), np.concatenate(y1_pred_all),
    )


def compute_pos_weights(train_runs: Sequence[PreprocRun]) -> Tuple[float, float, float]:
    yint = np.concatenate([r.y_int for r in train_runs], axis=0)
    pos = float((yint == 1).sum())
    neg = float((yint == 0).sum())
    w_int = (neg / max(pos, 1.0))

    mask = np.concatenate([r.mask for r in train_runs], axis=0).astype(bool)
    y0 = np.concatenate([r.y0_base for r in train_runs], axis=0)[mask]
    y1 = np.concatenate([r.y1_base for r in train_runs], axis=0)[mask]

    def w(y):
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return (neg / max(pos, 1.0))

    return w_int, w(y0), w(y1)


def train_deep(
    model: nn.Module,
    train_runs: Sequence[PreprocRun],
    test_runs: Sequence[PreprocRun],
    window: int,
    stride: int,
    batch: int,
    epochs: int,
    lr: float,
    wd: float,
    clip: float,
    lambda_ind: float,
    device: torch.device,
    use_amp: bool,
    num_workers: int,
    swap_prob: float,
    seed: int,
    smooth_k: int,
) -> None:
    ds_tr = PairWindowDataset(train_runs, window=window, stride=stride, swap_prob=swap_prob, seed=seed)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    w_int, w0, w1 = compute_pos_weights(train_runs)
    pos_w_int = torch.tensor([w_int], dtype=torch.float32, device=device)
    pos_w0 = torch.tensor([w0], dtype=torch.float32, device=device)
    pos_w1 = torch.tensor([w1], dtype=torch.float32, device=device)

    crit_int = nn.BCEWithLogitsLoss(pos_weight=pos_w_int)
    crit0 = nn.BCEWithLogitsLoss(pos_weight=pos_w0)
    crit1 = nn.BCEWithLogitsLoss(pos_weight=pos_w1)

    amp_enabled = (use_amp and device.type == "cuda")
    # Prefer the newer torch.amp API (avoids deprecation warnings).
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        def _autocast():
            return torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        def _autocast():
            return torch.cuda.amp.autocast(enabled=amp_enabled)


    def majority_smooth(y: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return y
        if k % 2 == 0:
            raise ValueError("--smooth-k must be odd")
        r = k // 2
        out = y.copy()
        for i in range(len(y)):
            a = max(0, i - r)
            b = min(len(y), i + r + 1)
            out[i] = 1 if (y[a:b].sum() > (b - a) / 2) else 0
        return out

    for epoch in range(1, epochs + 1):
        ds_tr.set_epoch(epoch)
        model.train()
        total = 0.0

        for xb, yintb, y0b, y1b, maskb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yintb = yintb.to(device, non_blocking=True).unsqueeze(-1)  # (B,W,1)
            y0b = y0b.to(device, non_blocking=True).unsqueeze(-1)
            y1b = y1b.to(device, non_blocking=True).unsqueeze(-1)
            maskb = maskb.to(device, non_blocking=True).unsqueeze(-1)  # 1 where non-interaction

            opt.zero_grad(set_to_none=True)

            with _autocast():
                li, l0, l1 = model(xb)  # each (B,W,1)
                loss_int = crit_int(li, yintb)

                # Masked individual losses
                # If mask sum is 0 (all interaction), skip ind losses gracefully.
                msum = maskb.sum()
                if msum.item() < 0.5:
                    loss0 = torch.tensor(0.0, device=device)
                    loss1 = torch.tensor(0.0, device=device)
                else:
                    loss0 = crit0(l0[maskb.bool()], y0b[maskb.bool()])
                    loss1 = crit1(l1[maskb.bool()], y1b[maskb.bool()])

                loss = loss_int + lambda_ind * 0.5 * (loss0 + loss1)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())

        # Evaluate
        yint_true, yint_pred, y0_true, y0_pred, y1_true, y1_pred = predict_full_runs(
            model, test_runs, window=window, stride=stride, device=device
        )

        if smooth_k > 1:
            # smooth interaction predictions across test runs
            yhat_sm = []
            start = 0
            for r in test_runs:
                T = r.y_int.shape[0]
                yhat_sm.append(majority_smooth(yint_pred[start:start + T], smooth_k))
                start += T
            yint_pred = np.concatenate(yhat_sm, axis=0)

        f1_int = f1_score(yint_true, yint_pred, pos_label=1, zero_division=0)
        f1_0 = f1_score(y0_true, y0_pred, pos_label=1, zero_division=0) if y0_true.size else 0.0
        f1_1 = f1_score(y1_true, y1_pred, pos_label=1, zero_division=0) if y1_true.size else 0.0

        avg_loss = total / max(1, len(dl_tr))
        print(f"[epoch {epoch:03d}] loss={avg_loss:.4f}  f1(interact)={f1_int:.4f}  f1(rear0)={f1_0:.4f}  f1(rear1)={f1_1:.4f}")

    print("Interaction report:")
    print(classification_report(yint_true, yint_pred, digits=4, zero_division=0))
    print("Rear0 report (non-interaction frames):")
    print(classification_report(y0_true, y0_pred, digits=4, zero_division=0))
    print("Rear1 report (non-interaction frames):")
    print(classification_report(y1_true, y1_pred, digits=4, zero_division=0))


# ----------------------------
# Run splitting (by run)
# ----------------------------

def split_runs(runs: List[PairRun], test_frac: float, seed: int) -> Tuple[List[PairRun], List[PairRun]]:
    if len(runs) < 2:
        r = runs[0]
        cut = int((1.0 - test_frac) * r.pos.shape[0])
        r1 = PairRun(name=r.name + "_train", times=r.times[:cut], pos=r.pos[:cut], beh=r.beh[:cut],
                    node_order=r.node_order, mouse_ids=r.mouse_ids)
        r2 = PairRun(name=r.name + "_test", times=r.times[cut:], pos=r.pos[cut:], beh=r.beh[cut:],
                    node_order=r.node_order, mouse_ids=r.mouse_ids)
        return [r1], [r2]
    rng = random.Random(seed)
    idx = list(range(len(runs)))
    rng.shuffle(idx)
    n_test = max(1, int(round(test_frac * len(runs))))
    test_idx = set(idx[:n_test])
    tr = [runs[i] for i in range(len(runs)) if i not in test_idx]
    te = [runs[i] for i in range(len(runs)) if i in test_idx]
    return tr, te


def preprocess_run(
    run: PairRun,
    tail_root: str,
    head: str,
    nose: str,
    use_vel: bool = True,
) -> PreprocRun:
    # Base ordering
    pos_pair_base = to_pair_frame(run.pos, run.node_order, tail_root=tail_root)
    X_base = make_pair_features(pos_pair_base, run.node_order, nose=nose, head=head, tail_root=tail_root, use_vel=use_vel)

    # Swapped ordering (swap mice indices, then recompute pair frame)
    pos_sw = run.pos.copy()
    pos_sw[:, 0], pos_sw[:, 1] = pos_sw[:, 1].copy(), pos_sw[:, 0].copy()
    pos_pair_swap = to_pair_frame(pos_sw, run.node_order, tail_root=tail_root)
    X_swap = make_pair_features(pos_pair_swap, run.node_order, nose=nose, head=head, tail_root=tail_root, use_vel=use_vel)

    y_int, y0, y1, mask = derive_multitask_labels(run.beh)
    # For swapped version, individual labels must swap
    y0_swap = y1.copy()
    y1_swap = y0.copy()

    return PreprocRun(
        name=run.name,
        X_base=X_base.astype(np.float32),
        X_swap=X_swap.astype(np.float32),
        y_int=y_int.astype(np.int64),
        y0_base=y0.astype(np.int64),
        y1_base=y1.astype(np.int64),
        y0_swap=y0_swap.astype(np.int64),
        y1_swap=y1_swap.astype(np.int64),
        mask=mask.astype(np.int64),
    )



# ----------------------------
# Ethogram export
# ----------------------------

def _safe_run_name(name: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    s = s.strip("_.-")
    return s or "run"


def _majority_smooth_binary(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y.copy()
    if k % 2 == 0:
        raise ValueError("--smooth-k must be odd")
    r = k // 2
    out = y.copy()
    for i in range(len(y)):
        a = max(0, i - r)
        b = min(len(y), i + r + 1)
        out[i] = 1 if (y[a:b].sum() > (b - a) / 2) else 0
    return out


def export_ethogram_ground_truth(out_dir: str, run: PairRun) -> str:
    """
    Writes a compact ground-truth ethogram per frame.

    Columns:
      frame,time,gt_interaction,gt_behavior0,gt_behavior1,gt_rear0,gt_rear1,gt_mask_non_interaction
    """
    os.makedirs(out_dir, exist_ok=True)
    rn = _safe_run_name(run.name)
    out_path = os.path.join(out_dir, f"ethogram_gt_{rn}.csv")

    y_int, y0, y1, mask = derive_multitask_labels(run.beh)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "time",
            "gt_interaction",
            "gt_behavior0", "gt_behavior1",
            "gt_rear0", "gt_rear1",
            "gt_mask_non_interaction",
        ])
        for i in range(run.pos.shape[0]):
            w.writerow([
                i, f"{float(run.times[i]):.6f}",
                int(y_int[i]),
                int(run.beh[i, 0]), int(run.beh[i, 1]),
                int(y0[i]), int(y1[i]),
                int(mask[i]),
            ])
    return out_path


def export_ethogram_prediction(
    out_dir: str,
    run: PairRun,
    int_prob: np.ndarray, int_pred: np.ndarray,
    rear0_prob: np.ndarray, rear0_pred: np.ndarray,
    rear1_prob: np.ndarray, rear1_pred: np.ndarray,
) -> str:
    """
    Writes predicted ethogram per frame.

    Applies hierarchy to emit per-mouse predicted behavior codes:
      if interaction==1 => behavior=2
      else behavior = 1 if rear==1 else 0
    """
    os.makedirs(out_dir, exist_ok=True)
    rn = _safe_run_name(run.name)
    out_path = os.path.join(out_dir, f"ethogram_pred_{rn}.csv")

    int_prob = np.asarray(int_prob).reshape(-1)
    int_pred = np.asarray(int_pred).reshape(-1).astype(np.int64)
    rear0_prob = np.asarray(rear0_prob).reshape(-1)
    rear0_pred = np.asarray(rear0_pred).reshape(-1).astype(np.int64)
    rear1_prob = np.asarray(rear1_prob).reshape(-1)
    rear1_pred = np.asarray(rear1_pred).reshape(-1).astype(np.int64)

    T = run.pos.shape[0]
    if not (len(int_prob) == len(int_pred) == len(rear0_prob) == len(rear0_pred) == len(rear1_prob) == len(rear1_pred) == T):
        raise ValueError("Ethogram export: prediction lengths do not match run length")

    # Hierarchical decoded behaviors
    beh0 = np.where(int_pred == 1, 2, np.where(rear0_pred == 1, 1, 0)).astype(np.int64)
    beh1 = np.where(int_pred == 1, 2, np.where(rear1_pred == 1, 1, 0)).astype(np.int64)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "time",
            "pred_interaction_prob", "pred_interaction",
            "pred_rear0_prob", "pred_rear0",
            "pred_rear1_prob", "pred_rear1",
            "pred_behavior0", "pred_behavior1",
        ])
        for i in range(T):
            w.writerow([
                i, f"{float(run.times[i]):.6f}",
                f"{float(int_prob[i]):.6f}", int(int_pred[i]),
                f"{float(rear0_prob[i]):.6f}", int(rear0_pred[i]),
                f"{float(rear1_prob[i]):.6f}", int(rear1_pred[i]),
                int(beh0[i]), int(beh1[i]),
            ])
    return out_path


@torch.no_grad()
def predict_probs_per_run(
    model: nn.Module,
    run: PreprocRun,
    window: int,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-frame probabilities for interaction, rear0, rear1 for ONE run:
      p_int(T,), p0(T,), p1(T,)
    Uses overlap-averaged logits (same as evaluation).
    """
    model.eval()
    X = run.X_base
    T = X.shape[0]

    logits_int = np.zeros((T,), dtype=np.float64)
    logits0 = np.zeros((T,), dtype=np.float64)
    logits1 = np.zeros((T,), dtype=np.float64)
    counts = np.zeros((T,), dtype=np.float64)

    def accumulate(s: int, e: int, li, l0, l1):
        logits_int[s:e] += li
        logits0[s:e] += l0
        logits1[s:e] += l1
        counts[s:e] += 1.0

    if T < window:
        xb = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
        li, l0, l1 = model(xb)
        li = li.squeeze(-1).detach().cpu().numpy()[0]
        l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
        l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
        accumulate(0, T, li, l0, l1)
    else:
        for s in range(0, T - window + 1, stride):
            xb = torch.tensor(X[s:s + window][None, :, :], dtype=torch.float32, device=device)
            li, l0, l1 = model(xb)
            li = li.squeeze(-1).detach().cpu().numpy()[0]
            l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
            l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
            accumulate(s, s + window, li, l0, l1)

        uncovered = counts < 0.5
        if uncovered.any():
            xb = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
            li, l0, l1 = model(xb)
            li = li.squeeze(-1).detach().cpu().numpy()[0]
            l0 = l0.squeeze(-1).detach().cpu().numpy()[0]
            l1 = l1.squeeze(-1).detach().cpu().numpy()[0]
            logits_int[uncovered] = li[uncovered]
            logits0[uncovered] = l0[uncovered]
            logits1[uncovered] = l1[uncovered]
            counts[uncovered] = 1.0

    logits_int /= counts
    logits0 /= counts
    logits1 /= counts

    p_int = 1.0 / (1.0 + np.exp(-logits_int))
    p0 = 1.0 / (1.0 + np.exp(-logits0))
    p1 = 1.0 / (1.0 + np.exp(-logits1))
    return p_int.astype(np.float64), p0.astype(np.float64), p1.astype(np.float64)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        "Pair behavior classifier (2 mice): multitask hierarchy with pair-frame coords + swap augmentation"
    )

    ap.add_argument("--coords-3d", nargs="+", required=True,
                    help="One or more coords_3d.csv paths from mouse_sim2.py (must include mouse_id).")
    ap.add_argument("--out-dir", default="train_behavior_pair_out",
                    help="Directory for reports and exported ethograms.")
    ap.add_argument("--no-ethograms", action="store_true",
                    help="Disable writing ethogram CSVs to --out-dir.")

    ap.add_argument("--ckpt-path", default=None,
                    help="Where to save a deep-model checkpoint (.pt). Default: <out-dir>/behavior_pair_<model>.pt")
    ap.add_argument("--no-save-ckpt", action="store_true",
                    help="Disable saving a deep-model checkpoint.")


    ap.add_argument("--model", default="lstm",
                    choices=["baseline_rule", "baseline_logreg", "lstm", "transformer", "xlstm"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")

    # Units / scaling
    ap.add_argument("--scale-mm", type=float, default=1.0,
                    help="Multiply all coordinates by this factor (set 10 if your CSV is in cm and you want mm).")

    # GPU performance switches
    ap.add_argument("--amp", action="store_true",
                    help="Enable automatic mixed precision (recommended on NVIDIA GPUs).")
    ap.add_argument("--tf32", action="store_true",
                    help="Enable TF32 matmul on NVIDIA GPUs (faster, slight precision loss).")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers (increase if CPU allows).")

    # Nodes
    ap.add_argument("--tail-root", default="tail_root")
    ap.add_argument("--head", default="head")
    ap.add_argument("--nose", default="nose_tip")

    # Split
    ap.add_argument("--test-frac", type=float, default=0.2)

    # Windowing
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)

    # Optim
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lambda-ind", type=float, default=1.0,
                    help="Weight for individual rearing losses (interaction loss weight is 1.0).")

    # Swap augmentation
    ap.add_argument("--swap-prob", type=float, default=0.5,
                    help="Probability to swap mouse identities in a training window (invariance augmentation).")

    # Smoothing for ethogram-style interaction predictions
    ap.add_argument("--smooth-k", type=int, default=1,
                    help="Odd window size for majority smoothing on interaction predictions (1 disables).")

    # LSTM / Transformer / xLSTM shared-ish
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-bidir", action="store_true")

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256)

    # xLSTM
    ap.add_argument("--xlstm-blocks", type=int, default=4)
    ap.add_argument("--xlstm-backend", default="cuda", choices=["native", "cuda"],
                    help="xLSTM sLSTM backend. Use 'native' if CUDA backend fails to compile.")
    ap.add_argument("--xlstm-kernel", type=int, default=4)

    # Baselines
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength.")
    ap.add_argument("--rule-dist", type=float, default=20.0,
                    help="Geometric rule: nose-nose distance threshold (in scaled units, e.g. mm).")
    ap.add_argument("--rule-face", type=float, default=0.4,
                    help="Geometric rule: facing dot-product threshold (0..1).")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    export_ethograms = (not args.no_ethograms)

    # Device
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    if args.smooth_k > 1 and args.smooth_k % 2 == 0:
        raise ValueError("--smooth-k must be odd")

    # Load runs
    runs: List[PairRun] = []
    base_order: Optional[List[str]] = None
    for p in args.coords_3d:
        r = load_coords_3d_csv_pair(p, node_order=base_order, scale_mm=args.scale_mm)
        if base_order is None:
            base_order = r.node_order
        runs.append(r)

    # Split by run
    train_runs_raw, test_runs_raw = split_runs(runs, test_frac=args.test_frac, seed=args.seed)
    print(f"runs: train={len(train_runs_raw)} test={len(test_runs_raw)} total={len(runs)}")

    if export_ethograms:
        for r in test_runs_raw:
            export_ethogram_ground_truth(args.out_dir, r)

    # Baseline rule uses raw PairRun
    if args.model == "baseline_rule":
        # Evaluate on test runs
        y_true_all = []
        y_pred_all = []
        for r in test_runs_raw:
            y_int_true, *_ = derive_multitask_labels(r.beh)
            yhat = geometric_rule_interaction(
                r, head=args.head, nose=args.nose, tail_root=args.tail_root,
                scale_mm=args.scale_mm, dist_thr=args.rule_dist, facing_thr=args.rule_face
            )
            if export_ethograms:
                # rule only predicts interaction; set rearing probs to NaN and rearing preds to 0
                T = r.pos.shape[0]
                export_ethogram_prediction(
                    args.out_dir, r,
                    int_prob=yhat.astype(np.float64), int_pred=yhat.astype(np.int64),
                    rear0_prob=np.full((T,), np.nan), rear0_pred=np.zeros((T,), dtype=np.int64),
                    rear1_prob=np.full((T,), np.nan), rear1_pred=np.zeros((T,), dtype=np.int64),
                )
            y_true_all.append(y_int_true)
            y_pred_all.append(yhat)
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        f1_int = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        print(f"[baseline_rule] f1(interact)={f1_int:.4f}")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        return

    # Preprocess runs into pair-frame features
    train_runs = [preprocess_run(r, tail_root=args.tail_root, head=args.head, nose=args.nose, use_vel=True)
                  for r in train_runs_raw]
    test_runs = [preprocess_run(r, tail_root=args.tail_root, head=args.head, nose=args.nose, use_vel=True)
                 for r in test_runs_raw]

    # Baseline logistic regression multitask (with optional ethogram export)
    if args.model == "baseline_logreg":
        # Fit 3 separate logistic regressions:
        #   - interaction on all frames
        #   - rear0 on non-interaction frames
        #   - rear1 on non-interaction frames
        Xtr = np.concatenate([r.X_base for r in train_runs], axis=0)
        Xte_all = np.concatenate([r.X_base for r in test_runs], axis=0)

        yint_tr = np.concatenate([r.y_int for r in train_runs], axis=0)
        yint_te = np.concatenate([r.y_int for r in test_runs], axis=0)

        mask_tr = np.concatenate([r.mask for r in train_runs], axis=0).astype(bool)
        mask_te = np.concatenate([r.mask for r in test_runs], axis=0).astype(bool)

        y0_tr = np.concatenate([r.y0_base for r in train_runs], axis=0)[mask_tr]
        y1_tr = np.concatenate([r.y1_base for r in train_runs], axis=0)[mask_tr]
        y0_te = np.concatenate([r.y0_base for r in test_runs], axis=0)[mask_te]
        y1_te = np.concatenate([r.y1_base for r in test_runs], axis=0)[mask_te]

        Xtr_mask = Xtr[mask_tr]
        Xte_mask = Xte_all[mask_te]

        clf_int = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]).fit(Xtr, yint_tr)

        clf0 = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]).fit(Xtr_mask, y0_tr)

        clf1 = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]).fit(Xtr_mask, y1_tr)

        # Evaluate (stacked)
        yhat_int = clf_int.predict(Xte_all)

        yhat0 = clf0.predict(Xte_mask)
        yhat1 = clf1.predict(Xte_mask)

        if args.smooth_k > 1:
            yhat_sm = []
            start = 0
            for r in test_runs:
                T = r.y_int.shape[0]
                yhat_sm.append(_majority_smooth_binary(yhat_int[start:start + T], args.smooth_k))
                start += T
            yhat_int = np.concatenate(yhat_sm, axis=0)

        f1_int = f1_score(yint_te, yhat_int, pos_label=1, zero_division=0)
        f1_0 = f1_score(y0_te, yhat0, pos_label=1, zero_division=0)
        f1_1 = f1_score(y1_te, yhat1, pos_label=1, zero_division=0)
        print(f"[baseline_logreg] f1(interact)={f1_int:.4f}  f1(rear0)={f1_0:.4f}  f1(rear1)={f1_1:.4f}")
        print("Interaction report:")
        print(classification_report(yint_te, yhat_int, digits=4, zero_division=0))
        print("Rear0 report (non-interaction frames):")
        print(classification_report(y0_te, yhat0, digits=4, zero_division=0))
        print("Rear1 report (non-interaction frames):")
        print(classification_report(y1_te, yhat1, digits=4, zero_division=0))

        if export_ethograms:
            # Export per-run predictions (not masked; hierarchy decides final behaviors)
            for raw, pr in zip(test_runs_raw, test_runs):
                X = pr.X_base
                p_int = clf_int.predict_proba(X)[:, 1]
                int_pred = (p_int >= 0.5).astype(np.int64)
                if args.smooth_k > 1:
                    int_pred = _majority_smooth_binary(int_pred, args.smooth_k)

                p0 = clf0.predict_proba(X)[:, 1]
                p1 = clf1.predict_proba(X)[:, 1]
                rear0_pred = (p0 >= 0.5).astype(np.int64)
                rear1_pred = (p1 >= 0.5).astype(np.int64)

                export_ethogram_prediction(
                    args.out_dir, raw,
                    int_prob=p_int, int_pred=int_pred,
                    rear0_prob=p0, rear0_pred=rear0_pred,
                    rear1_prob=p1, rear1_pred=rear1_pred,
                )
        return

    # Determine input dim
    in_dim = train_runs[0].X_base.shape[1]

    # Build model
    if args.model == "lstm":
        backbone = LSTMBackbone(
            in_dim=in_dim,
            hidden=args.hidden,
            layers=args.layers,
            bidir=not args.no_bidir,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        backbone = TransformerBackbone(
            in_dim=in_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            ff=args.ff,
            dropout=args.dropout,
        )
    elif args.model == "xlstm":
        backbone = xLSTMBackbone(
            in_dim=in_dim,
            d_model=args.d_model,
            num_blocks=args.xlstm_blocks,
            context_length=args.window,
            backend=args.xlstm_backend,
            num_heads=args.nhead,
            conv1d_kernel_size=args.xlstm_kernel,
        )
    else:
        raise ValueError(args.model)

    model = MultiTaskHead(backbone, out_dim=backbone.out_dim)

    train_deep(
        model=model,
        train_runs=train_runs,
        test_runs=test_runs,
        window=args.window,
        stride=args.stride,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        clip=args.clip,
        lambda_ind=args.lambda_ind,
        device=device,
        use_amp=args.amp,
        num_workers=args.num_workers,
        swap_prob=args.swap_prob,
        seed=args.seed,
        smooth_k=args.smooth_k,
    )



    # Save deep-model checkpoint for later inference (predict_behavior_pair.py).
    if (args.model in {"lstm", "transformer", "xlstm"}) and (not args.no_save_ckpt):
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_path = os.path.join(args.out_dir, f"behavior_pair_{args.model}.pt")

        if args.model == "lstm":
            model_cfg = {
                "hidden": int(args.hidden),
                "layers": int(args.layers),
                "bidir": bool(not args.no_bidir),
                "dropout": float(args.dropout),
            }
        elif args.model == "transformer":
            model_cfg = {
                "d_model": int(args.d_model),
                "nhead": int(args.nhead),
                "layers": int(args.layers),
                "ff": int(args.ff),
                "dropout": float(args.dropout),
            }
        else:  # xlstm
            model_cfg = {
                "d_model": int(args.d_model),
                "nhead": int(args.nhead),
                "num_blocks": int(args.xlstm_blocks),
                "context_length": int(args.window),
                "backend": str(args.xlstm_backend),
                "conv1d_kernel_size": int(args.xlstm_kernel),
            }

        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ckpt = {
            "model_type": str(args.model),
            "model_cfg": model_cfg,
            "in_dim": int(in_dim),
            "model_state": state_cpu,
            "node_order": list(base_order or []),
            "scale_mm": float(args.scale_mm),
            "tail_root": str(args.tail_root),
            "head": str(args.head),
            "nose": str(args.nose),
            "use_vel": True,
            "window": int(args.window),
            "stride": int(args.stride),
            "smooth_k": int(args.smooth_k),
        }
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")


    if export_ethograms:
        for raw, pr in zip(test_runs_raw, test_runs):
            p_int, p0, p1 = predict_probs_per_run(model, pr, window=args.window, stride=args.stride, device=device)
            int_pred = (p_int >= 0.5).astype(np.int64)
            if args.smooth_k > 1:
                int_pred = _majority_smooth_binary(int_pred, args.smooth_k)

            rear0_pred = (p0 >= 0.5).astype(np.int64)
            rear1_pred = (p1 >= 0.5).astype(np.int64)

            export_ethogram_prediction(
                args.out_dir, raw,
                int_prob=p_int, int_pred=int_pred,
                rear0_prob=p0, rear0_pred=rear0_pred,
                rear1_prob=p1, rear1_pred=rear1_pred,
            )
        print(f"Ethograms written to: {args.out_dir}")



if __name__ == "__main__":
    main()
