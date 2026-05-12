#!/usr/bin/env python3
"""
overlay_dlc_on_videos_v2.py

Reads a DeepLabCut-style long-format CSV (dlc_long.csv) with columns:
  frame, camera, individual, bodypart, x, y, likelihood
(Extra columns like scorer are ignored.)

Overlays body-part detections (and optional skeleton edges from a template [EDGES] section)
onto three input videos (top/front/side) and writes three annotated output videos.

Additionally, if --joinedVideo is provided, writes a single mosaic video of the SAME
resolution as the input videos:

  +-------------------+-------------------+
  | (empty)           | top (downsampled) |
  +-------------------+-------------------+
  | side (downsampled)| front (downsampled)|
  +-------------------+-------------------+

Downsampling uses cv2.INTER_AREA to reduce aliasing.

Requires: opencv (cv2), numpy, pandas
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2

# Optional: Segment Anything (SAM) for segmentation-based visualization
# Install per official repo: pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    import torch  # type: ignore
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False


# -----------------------------
# Template edges (optional)
# -----------------------------
def parse_edges_from_template(path: str | Path) -> List[Tuple[str, str]]:
    """Parse [EDGES] section with lines: 'nodeA nodeB'."""
    p = Path(path)
    edges: List[Tuple[str, str]] = []
    in_edges = False
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper() == "[EDGES]":
            in_edges = True
            continue
        if s.startswith("[") and s.endswith("]") and s.upper() != "[EDGES]":
            in_edges = False
        if in_edges:
            parts = s.split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


# -----------------------------
# Drawing helpers
# -----------------------------
def make_color_map(individuals: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Deterministic BGR colors (OpenCV)."""
    colors = [
        (255, 255, 255),  # white
        (80, 220, 80),    # green-ish
        (220, 80, 80),
        (80, 80, 220),
        (220, 220, 80),
        (220, 80, 220),
    ]
    return {ind: colors[i % len(colors)] for i, ind in enumerate(individuals)}


def overlay_frame(
    frame_bgr: np.ndarray,
    rows: pd.DataFrame,
    edges: Optional[List[Tuple[str, str]]],
    likelihood_min: float,
    draw_labels: bool,
    label_scale: float,
    label_thickness: int,
) -> np.ndarray:
    """Overlay detections for a single frame (already filtered to one camera)."""
    if rows.empty:
        return frame_bgr

    inds = sorted(rows["individual"].dropna().astype(str).unique().tolist())
    cmap = make_color_map(inds)

    coords: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for r in rows.itertuples(index=False):
        ind = str(r.individual)
        bp = str(r.bodypart)
        x = float(r.x) if pd.notna(r.x) else np.nan
        y = float(r.y) if pd.notna(r.y) else np.nan
        lik = float(r.likelihood) if pd.notna(r.likelihood) else 0.0
        coords[(ind, bp)] = (x, y, lik)

    # skeleton
    if edges:
        for ind in inds:
            col = cmap[ind]
            for a, b in edges:
                ka, kb = (ind, a), (ind, b)
                if ka not in coords or kb not in coords:
                    continue
                xa, ya, la = coords[ka]
                xb, yb, lb = coords[kb]
                if la < likelihood_min or lb < likelihood_min:
                    continue
                if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
                    continue
                cv2.line(frame_bgr, (int(xa), int(ya)), (int(xb), int(yb)), col, 2, cv2.LINE_AA)

    # points
    for ind in inds:
        col = cmap[ind]
        for (i_ind, bp), (x, y, lik) in coords.items():
            if i_ind != ind:
                continue
            if lik < likelihood_min:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cv2.circle(frame_bgr, (int(x), int(y)), 4, col, -1, cv2.LINE_AA)
            if draw_labels:
                cv2.putText(
                    frame_bgr,
                    f"{ind}:{bp}",
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(label_scale),
                    col,
                    int(label_thickness),
                    cv2.LINE_AA,
                )
    return frame_bgr


# -----------------------------
# Per-video overlay
# -----------------------------
def overlay_video(
    in_path: str | Path,
    out_path: str | Path,
    df_cam: pd.DataFrame,
    edges: Optional[List[Tuple[str, str]]],
    likelihood_min: float,
    draw_labels: bool,
    label_scale: float,
    label_thickness: int,
    force_fps: Optional[float],
) -> Tuple[int, int, float, int]:
    """Overlay detections onto a single video. Returns (w,h,fps,num_frames_written)."""
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    fps = float(force_fps) if force_fps is not None else (fps_in if fps_in > 1e-6 else 10.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    df_cam = df_cam.sort_values("frame").reset_index(drop=True)
    frames_arr = df_cam["frame"].to_numpy(dtype=np.int64)
    nrows = len(df_cam)
    idx = 0

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        while idx < nrows and frames_arr[idx] < frame_idx:
            idx += 1
        start = idx
        while idx < nrows and frames_arr[idx] == frame_idx:
            idx += 1

        rows = df_cam.iloc[start:idx] if idx > start else df_cam.iloc[0:0]
        out_frame = overlay_frame(frame, rows, edges, likelihood_min, draw_labels, label_scale, label_thickness)
        writer.write(out_frame)
        frame_idx += 1
        written += 1

    cap.release()
    writer.release()
    return w, h, fps, written


# -----------------------------
# Joined mosaic writer
# -----------------------------
def overlay_three_and_join(
    top_in: str | Path, front_in: str | Path, side_in: str | Path,
    joined_out: str | Path,
    df_top: pd.DataFrame, df_front: pd.DataFrame, df_side: pd.DataFrame,
    edges: Optional[List[Tuple[str, str]]],
    likelihood_min: float,
    draw_labels: bool,
    label_scale: float,
    label_thickness: int,
    force_fps: Optional[float],
) -> None:
    top_cap = cv2.VideoCapture(str(top_in))
    front_cap = cv2.VideoCapture(str(front_in))
    side_cap = cv2.VideoCapture(str(side_in))
    if not top_cap.isOpened():
        raise RuntimeError(f"Could not open top input video: {top_in}")
    if not front_cap.isOpened():
        raise RuntimeError(f"Could not open front input video: {front_in}")
    if not side_cap.isOpened():
        raise RuntimeError(f"Could not open side input video: {side_in}")

    w = int(top_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(top_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(top_cap.get(cv2.CAP_PROP_FPS))
    fps = float(force_fps) if force_fps is not None else (fps_in if fps_in > 1e-6 else 10.0)

    # quadrant sizes (handle odd sizes by assigning remainder to right/bottom)
    w_left = w // 2
    w_right = w - w_left
    h_top = h // 2
    h_bottom = h - h_top

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(joined_out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open joined output video: {joined_out}")

    def prep(df_cam: pd.DataFrame):
        df_cam = df_cam.sort_values("frame").reset_index(drop=True)
        return df_cam, df_cam["frame"].to_numpy(dtype=np.int64)

    df_top, fr_top = prep(df_top)
    df_front, fr_front = prep(df_front)
    df_side, fr_side = prep(df_side)
    idx_top = idx_front = idx_side = 0

    def slice_rows(df_cam, fr_arr, idx, frame_idx):
        n = len(fr_arr)
        while idx < n and fr_arr[idx] < frame_idx:
            idx += 1
        start = idx
        while idx < n and fr_arr[idx] == frame_idx:
            idx += 1
        rows = df_cam.iloc[start:idx] if idx > start else df_cam.iloc[0:0]
        return rows, idx

    frame_idx = 0
    while True:
        ok_t, frame_t = top_cap.read()
        ok_f, frame_f = front_cap.read()
        ok_s, frame_s = side_cap.read()
        if not (ok_t and ok_f and ok_s):
            break

        # If other videos differ in size, resize to top size BEFORE overlay (best effort).
        if frame_f.shape[1] != w or frame_f.shape[0] != h:
            frame_f = cv2.resize(frame_f, (w, h), interpolation=cv2.INTER_LINEAR)
        if frame_s.shape[1] != w or frame_s.shape[0] != h:
            frame_s = cv2.resize(frame_s, (w, h), interpolation=cv2.INTER_LINEAR)

        rows_t, idx_top = slice_rows(df_top, fr_top, idx_top, frame_idx)
        rows_f, idx_front = slice_rows(df_front, fr_front, idx_front, frame_idx)
        rows_s, idx_side = slice_rows(df_side, fr_side, idx_side, frame_idx)

        frame_t = overlay_frame(frame_t, rows_t, edges, likelihood_min, draw_labels, label_scale, label_thickness)
        frame_f = overlay_frame(frame_f, rows_f, edges, likelihood_min, draw_labels, label_scale, label_thickness)
        frame_s = overlay_frame(frame_s, rows_s, edges, likelihood_min, draw_labels, label_scale, label_thickness)

        # Downsample professionally for each quadrant (INTER_AREA)
        top_small = cv2.resize(frame_t, (w_right, h_top), interpolation=cv2.INTER_AREA)
        front_small = cv2.resize(frame_f, (w_right, h_bottom), interpolation=cv2.INTER_AREA)
        side_small = cv2.resize(frame_s, (w_left, h_bottom), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # top-right
        canvas[0:h_top, w_left:w] = top_small
        # bottom-left
        canvas[h_top:h, 0:w_left] = side_small
        # bottom-right
        canvas[h_top:h, w_left:w] = front_small

        writer.write(canvas)
        frame_idx += 1

    top_cap.release()
    front_cap.release()
    side_cap.release()
    writer.release()




# -----------------------------
# Color-based identity correction (2 mice)
# -----------------------------
def parse_bgr_triplet(s: str) -> np.ndarray:
    """Parse 'b,g,r' or 'b g r' into float64 array shape (3,)."""
    if s is None:
        raise ValueError("Empty BGR string")
    t = str(s).strip().replace(",", " ").split()
    if len(t) != 3:
        raise ValueError(f"Expected 3 values for BGR, got: {s!r}")
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def _kmeans2(points: np.ndarray, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Small kmeans for k=2. Returns (centers (2,3), labels (N,))."""
    if len(points) < 2:
        raise ValueError("Need >=2 points for kmeans2")
    pts = points.astype(np.float64)

    # init: farthest pair heuristic
    p0 = pts[0]
    d = np.sum((pts - p0) ** 2, axis=1)
    c1 = pts[int(np.argmax(d))]
    d2 = np.sum((pts - c1) ** 2, axis=1)
    c0 = pts[int(np.argmax(d2))]
    centers = np.stack([c0, c1], axis=0).astype(np.float64)

    labels = np.zeros(len(pts), dtype=np.int32)
    for _ in range(iters):
        dist0 = np.sum((pts - centers[0]) ** 2, axis=1)
        dist1 = np.sum((pts - centers[1]) ** 2, axis=1)
        labels = (dist1 < dist0).astype(np.int32)
        for k in (0, 1):
            sel = pts[labels == k]
            if len(sel) > 0:
                centers[k] = np.mean(sel, axis=0)

    dist0 = np.sum((pts - centers[0]) ** 2, axis=1)
    dist1 = np.sum((pts - centers[1]) ** 2, axis=1)
    labels = (dist1 < dist0).astype(np.int32)
    return centers, labels


def _pick_separation_frames(df_cam: pd.DataFrame, min_lik: float, k: int) -> List[int]:
    """Pick frames where the two labeled individuals are far apart (centroid distance)."""
    d = df_cam[df_cam["likelihood"] >= min_lik].copy()
    if d.empty:
        return []
    inds = sorted(d["individual"].astype(str).unique().tolist())
    if len(inds) != 2:
        return []
    mapping = {inds[0]: 0, inds[1]: 1}
    d["mid_tmp"] = d["individual"].astype(str).map(mapping).astype(int)
    g = d.groupby(["frame", "mid_tmp"])[["x", "y"]].mean().reset_index()
    g0 = g[g["mid_tmp"] == 0].set_index("frame")
    g1 = g[g["mid_tmp"] == 1].set_index("frame")
    frames = sorted(list(set(g0.index).intersection(set(g1.index))))
    if not frames:
        return []
    dist = []
    for fr in frames:
        dx = float(g0.loc[fr, "x"] - g1.loc[fr, "x"])
        dy = float(g0.loc[fr, "y"] - g1.loc[fr, "y"])
        dist.append((int(fr), (dx*dx + dy*dy) ** 0.5))
    dist.sort(key=lambda t: t[1], reverse=True)
    return [fr for fr, _ in dist[:k]]


def _median_patch(frame_bgr: np.ndarray, x: float, y: float, r: int) -> Optional[np.ndarray]:
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    H, W = frame_bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(W, xi + r + 1)
    y0, y1 = max(0, yi - r), min(H, yi + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    pix = patch.reshape(-1, 3).astype(np.float64)
    return np.median(pix, axis=0)


def _assign_color_id(
    frame_bgr: np.ndarray,
    x: float, y: float,
    r: int,
    proto0_bgr: np.ndarray,  # mouse0
    proto1_bgr: np.ndarray,  # mouse1
    dist_thresh: float,
    min_frac: float,
    min_pixels: int,
    dominance: float,
) -> Tuple[int, float]:
    """Return (assigned_id in {0,1,-1}, confidence)."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return -1, 0.0
    H, W = frame_bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(W, xi + r + 1)
    y0, y1 = max(0, yi - r), min(H, yi + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return -1, 0.0
    pix = patch.reshape(-1, 3).astype(np.float64)
    n = pix.shape[0]
    if n < min_pixels:
        return -1, 0.0
    thr2 = float(dist_thresh) ** 2
    d0 = np.sum((pix - proto0_bgr) ** 2, axis=1)
    d1 = np.sum((pix - proto1_bgr) ** 2, axis=1)
    c0 = int(np.sum(d0 <= thr2))
    c1 = int(np.sum(d1 <= thr2))
    best = 0 if c0 >= c1 else 1
    bestc = max(c0, c1)
    otherc = min(c0, c1)
    frac = bestc / max(1, n)
    if frac < min_frac:
        return -1, 0.0
    if bestc < otherc * (1.0 + dominance) and (bestc - otherc) < 5:
        return -1, 0.0
    conf = (bestc - otherc) / max(1, n)
    return best, float(conf)


def _paw_centroid(rows: pd.DataFrame, paw_parts: List[str], min_lik: float) -> Optional[Tuple[float, float]]:
    sub = rows[(rows["bodypart"].astype(str).isin(paw_parts)) & (rows["likelihood"] >= min_lik)]
    if len(sub) < 2:
        return None
    return float(sub["x"].mean()), float(sub["y"].mean())


def _fallback_centroid(rows: pd.DataFrame, parts: List[str], min_lik: float) -> Optional[Tuple[float, float]]:
    sub = rows[(rows["bodypart"].astype(str).isin(parts)) & (rows["likelihood"] >= min_lik)]
    if len(sub) < 2:
        return None
    return float(sub["x"].mean()), float(sub["y"].mean())


def estimate_fur_prototypes_from_paw_cog(
    df_cam: pd.DataFrame,
    video_path: str | Path,
    sep_frames: List[int],
    patch_radius: int,
    calib_likelihood: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mouse0_bgr, mouse1_bgr) using center-of-gravity of paws (or fallback core parts).
    Color samples (BGR) from BOTH labeled individuals are clustered into 2 clusters in color space,
    disregarding IDs. Cluster with larger (R-B) is assigned to mouse0 (brownish), other to mouse1.
    """
    paw_parts = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
    fallback_parts = ["head", "tail_root", "nose_tip", "left_ear_tip", "right_ear_tip"]

    inds = sorted(df_cam["individual"].astype(str).unique().tolist())
    if len(inds) != 2:
        raise ValueError(f"Expected 2 individuals, got: {inds}")

    df_cam = df_cam.copy()
    df_cam["frame"] = df_cam["frame"].astype(int)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for calibration: {video_path}")

    df_cam = df_cam.sort_values("frame").reset_index(drop=True)
    frames_arr = df_cam["frame"].to_numpy(dtype=np.int64)
    nrows = len(df_cam)
    idx = 0

    sep_set = set(sep_frames)
    max_fr = max(sep_frames) if sep_frames else -1
    samples = []

    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_fr >= 0 and fr > max_fr:
            break

        if fr in sep_set:
            while idx < nrows and frames_arr[idx] < fr:
                idx += 1
            start = idx
            while idx < nrows and frames_arr[idx] == fr:
                idx += 1
            if idx > start:
                rows_fr = df_cam.iloc[start:idx]
                for ind in inds:
                    rows_ind = rows_fr[rows_fr["individual"].astype(str) == ind]
                    cog = _paw_centroid(rows_ind, paw_parts, calib_likelihood)
                    if cog is None:
                        cog = _fallback_centroid(rows_ind, fallback_parts, calib_likelihood)
                    if cog is None:
                        continue
                    c = _median_patch(frame, cog[0], cog[1], patch_radius)
                    if c is not None and np.all(np.isfinite(c)):
                        samples.append(c)

        fr += 1

    cap.release()

    if len(samples) < 20:
        raise RuntimeError("Too few color samples for calibration (need >=20).")

    pts = np.stack(samples, axis=0)
    _, labels = _kmeans2(pts, iters=30)

    protoA = np.median(pts[labels == 0], axis=0)
    protoB = np.median(pts[labels == 1], axis=0)

    if (protoA[2] - protoA[0]) >= (protoB[2] - protoB[0]):
        return protoA.astype(np.float64), protoB.astype(np.float64)
    else:
        return protoB.astype(np.float64), protoA.astype(np.float64)


def correct_identities_for_camera(
    df_cam: pd.DataFrame,
    video_path: str | Path,
    sep_likelihood: float,
    calib_frames: int,
    calib_likelihood: float,
    vote_likelihood: float,
    patch_radius: int,
    color_dist: float,
    min_fur_frac: float,
    min_pixels: int,
    dominance: float,
    drop_on_conflict: bool,
    manual_mouse0_bgr: Optional[np.ndarray] = None,
    manual_mouse1_bgr: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Step 3: per-frame majority check against DLC IDs -> swap mapping if needed.
    Step 4: per-part corrections; if conflict and --drop-on-conflict, drop leaving part.
    """
    df_cam = df_cam.copy()
    df_cam["frame"] = df_cam["frame"].astype(int)

    inds = sorted(df_cam["individual"].astype(str).unique().tolist())
    if len(inds) != 2:
        print(f"[autoCorrect] Expected 2 individuals, found {inds}. Skipping correction.")
        return df_cam

    # Only used for comparing to DLC IDs; doesn't assume these correspond to true mouse0/1.
    ind_to_mid = {inds[0]: 0, inds[1]: 1}
    df_cam["mid"] = df_cam["individual"].astype(str).map(ind_to_mid).astype(int)

    # Step 1: prototypes
    if manual_mouse0_bgr is not None and manual_mouse1_bgr is not None:
        proto0_bgr = manual_mouse0_bgr.astype(np.float64)
        proto1_bgr = manual_mouse1_bgr.astype(np.float64)
        print(f"[autoCorrect] Using MANUAL prototypes (BGR): mouse0~{proto0_bgr.round(1).tolist()} mouse1~{proto1_bgr.round(1).tolist()}")
    else:
        sep_frames = _pick_separation_frames(df_cam, min_lik=sep_likelihood, k=calib_frames)
        if len(sep_frames) < 3:
            print("[autoCorrect] Not enough separated frames found; skipping correction.")
            return df_cam.drop(columns=["mid"], errors="ignore")
        proto0_bgr, proto1_bgr = estimate_fur_prototypes_from_paw_cog(
            df_cam=df_cam.drop(columns=["mid"], errors="ignore"),
            video_path=video_path,
            sep_frames=sep_frames,
            patch_radius=patch_radius,
            calib_likelihood=calib_likelihood,
        )
        print(f"[autoCorrect] Estimated prototypes (BGR): mouse0(brownish)~{proto0_bgr.round(1).tolist()} mouse1(blueish)~{proto1_bgr.round(1).tolist()}")

    # Step 2: assign color_id per row
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for correction: {video_path}")

    df_cam = df_cam.sort_values("frame").reset_index(drop=True)
    frames_arr = df_cam["frame"].to_numpy(dtype=np.int64)
    nrows = len(df_cam)

    df_cam["color_id"] = -1
    df_cam["color_conf"] = 0.0

    idx = 0
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while idx < nrows and frames_arr[idx] < fr:
            idx += 1
        start = idx
        while idx < nrows and frames_arr[idx] == fr:
            idx += 1
        if idx > start:
            sub = df_cam.iloc[start:idx]
            for ridx, r in zip(range(start, idx), sub.itertuples(index=False)):
                if float(r.likelihood) < vote_likelihood:
                    continue
                cid, conf = _assign_color_id(
                    frame, float(r.x), float(r.y),
                    patch_radius,
                    proto0_bgr, proto1_bgr,
                    color_dist, min_fur_frac, min_pixels, dominance
                )
                if cid != -1:
                    df_cam.at[ridx, "color_id"] = cid
                    df_cam.at[ridx, "color_conf"] = conf
        fr += 1

    cap.release()

    votes = df_cam[(df_cam["color_id"] >= 0) & (df_cam["likelihood"] >= vote_likelihood)]
    if votes.empty:
        print("[autoCorrect] No usable color votes; skipping correction.")
        return df_cam.drop(columns=["mid"], errors="ignore")

    # Step 3: frame swap decision
    def score_frame(x: pd.DataFrame) -> pd.Series:
        mid = x["mid"].to_numpy(dtype=int)
        cid = x["color_id"].to_numpy(dtype=int)
        nos = int(np.sum(cid == mid))
        sw = int(np.sum(cid == (1 - mid)))
        return pd.Series({"noswap": nos, "swap": sw})

    scores = votes.groupby("frame").apply(score_frame).reset_index()
    swap_frames = set(scores[scores["swap"] > scores["noswap"]]["frame"].astype(int).tolist())
    if swap_frames:
        print(f"[autoCorrect] Frame-level swaps: {len(swap_frames)} frames.")

    df_cam["mid_corr"] = df_cam["mid"].astype(int)
    m = df_cam["frame"].isin(swap_frames)
    df_cam.loc[m, "mid_corr"] = 1 - df_cam.loc[m, "mid_corr"]

    # Step 4: per-bodypart correction
    keep = np.ones(len(df_cam), dtype=bool)
    for (fr, bp), grp in df_cam.groupby(["frame", "bodypart"]):
        rows = {}
        for mid in (0, 1):
            sub = grp[grp["mid_corr"] == mid]
            if sub.empty:
                continue
            rows[mid] = sub.sort_values("likelihood", ascending=False).iloc[0]
        if not rows:
            continue

        if 0 in rows and 1 in rows:
            r0, r1 = rows[0], rows[1]
            if int(r0["color_id"]) == 1 and int(r1["color_id"]) == 0:
                df_cam.at[r0.name, "mid_corr"] = 1
                df_cam.at[r1.name, "mid_corr"] = 0
                continue

        for mid, r in rows.items():
            cid = int(r["color_id"])
            if cid < 0 or cid == mid:
                continue
            target = cid
            if target not in rows:
                df_cam.at[r.name, "mid_corr"] = target
            else:
                if drop_on_conflict:
                    keep[r.name] = False

    df_cam = df_cam[keep].copy()
    df_cam["individual"] = df_cam["mid_corr"].astype(int).map(lambda m: f"mouse{m}")

    return df_cam.drop(columns=["mid", "mid_corr"], errors="ignore")



# -----------------------------
# SAM helpers (segmentation visualization)
# -----------------------------

def _kmeans2_xy(points_xy: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    # KMeans k=2 in 2D. Returns (centers (2,2), labels (N,))
    if len(points_xy) < 2:
        raise ValueError("Need >=2 points for kmeans2_xy")
    pts = points_xy.astype(np.float64)
    p0 = pts[0]
    d = np.sum((pts - p0) ** 2, axis=1)
    c1 = pts[int(np.argmax(d))]
    d2 = np.sum((pts - c1) ** 2, axis=1)
    c0 = pts[int(np.argmax(d2))]
    centers = np.stack([c0, c1], axis=0).astype(np.float64)

    labels = np.zeros(len(pts), dtype=np.int32)
    for _ in range(iters):
        dist0 = np.sum((pts - centers[0]) ** 2, axis=1)
        dist1 = np.sum((pts - centers[1]) ** 2, axis=1)
        labels = (dist1 < dist0).astype(np.int32)
        for k in (0, 1):
            sel = pts[labels == k]
            if len(sel) > 0:
                centers[k] = np.mean(sel, axis=0)

    dist0 = np.sum((pts - centers[0]) ** 2, axis=1)
    dist1 = np.sum((pts - centers[1]) ** 2, axis=1)
    labels = (dist1 < dist0).astype(np.int32)
    return centers, labels


def _two_mouse_prompts_from_rows(rows: pd.DataFrame, min_lik: float, use_dlc_ids: bool) -> Optional[np.ndarray]:
    # Return point prompts (2,2) in pixel coords (x,y) for the two mice.
    rows = rows[rows["likelihood"] >= min_lik]
    if len(rows) < 4:
        return None

    if use_dlc_ids:
        inds = sorted(rows["individual"].dropna().astype(str).unique().tolist())
        if len(inds) < 2:
            return None
        inds = inds[:2]
        c = []
        for ind in inds:
            sub = rows[rows["individual"].astype(str) == ind]
            if len(sub) < 2:
                return None
            c.append([float(sub["x"].mean()), float(sub["y"].mean())])
        return np.array(c, dtype=np.float32)

    pts = rows[["x", "y"]].to_numpy(dtype=np.float64)
    try:
        centers, _ = _kmeans2_xy(pts, iters=25)
    except Exception:
        return None
    return centers.astype(np.float32)


def _resolve_sam_device(device: str) -> str:
    d = str(device).strip().lower()
    if d == 'auto':
        return 'cuda' if (_HAS_SAM and torch.cuda.is_available()) else 'cpu'
    if d == 'cuda' and (not _HAS_SAM or not torch.cuda.is_available()):
        print('[SAM] CUDA requested but not available; falling back to CPU.')
        return 'cpu'
    return d


def _sam_load_predictor(checkpoint: str, model_type: str, device: str):
    if not _HAS_SAM:
        raise RuntimeError(
            "segment_anything/torch not available. Install per https://github.com/facebookresearch/segment-anything "
            "and ensure torch is installed."
        )
    model_type = str(model_type)
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {model_type}. Available: {sorted(sam_model_registry.keys())}")
    resolved_device = _resolve_sam_device(device)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=resolved_device)
    predictor = SamPredictor(sam)
    if resolved_device == 'cuda' and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = 'unknown CUDA device'
        print(f"[SAM] Using CUDA device: {gpu_name}")
    else:
        print(f"[SAM] Using device: {resolved_device}")
    return predictor

def _robust_sigma_from_mad(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    med = np.median(arr, axis=axis)
    mad = np.median(np.abs(arr - med), axis=axis)
    sigma = 1.4826 * mad
    return np.maximum(sigma, 1.0)


def _mask_area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_prompt_inside(mask: Optional[np.ndarray], prompt_xy: np.ndarray) -> Optional[bool]:
    if mask is None:
        return None
    if mask.size == 0:
        return None
    x = int(round(float(prompt_xy[0])))
    y = int(round(float(prompt_xy[1])))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def _encode_mask_for_json(mask: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Compact encoding for debug JSON.

    Stores a boolean mask as packed bits, zlib-compressed, base64-encoded.
    Intended for a small number of debug frames; do not enable for full videos.
    """
    if mask is None:
        return None
    if mask.size == 0:
        return None
    packed = np.packbits(mask.astype(np.uint8).reshape(-1), bitorder='big')
    blob = zlib.compress(packed.tobytes(), level=6)
    b64 = base64.b64encode(blob).decode('ascii')
    return {
        'encoding': 'packbits_zlib_base64',
        'shape': [int(mask.shape[0]), int(mask.shape[1])],
        'bitorder': 'big',
        'data': b64,
    }


def _choice_log_entry(choice: Optional[Dict[str, Any]], prompt_xy: np.ndarray, stats: Dict[str, Any], include_mask: bool) -> Optional[Dict[str, Any]]:
    if choice is None:
        return None
    mask = choice.get('mask', None)
    area = int(choice.get('area', 0) or 0)
    area_mean = float(stats.get('area_mean', 1.0) or 1.0)
    area_ratio = float(area) / area_mean if area_mean > 1e-9 else 0.0
    centroid = _mask_centroid(mask) if mask is not None else None
    prompt_inside = _mask_prompt_inside(mask, prompt_xy)
    out: Dict[str, Any] = {
        'passes': bool(choice.get('passes', False)),
        'penalty': None if 'penalty' not in choice else float(choice.get('penalty', 0.0)),
        'sam_score': None if 'sam_score' not in choice else float(choice.get('sam_score', 0.0)),
        'area': area,
        'area_ratio': float(area_ratio),
        'bbox': _mask_bbox(mask) if mask is not None else None,
        'centroid_xy': None if centroid is None else [float(centroid[0]), float(centroid[1])],
        'prompt_inside': None if prompt_inside is None else bool(prompt_inside),
    }
    if include_mask:
        out['mask'] = _encode_mask_for_json(mask)
    return out


def _clone_choice_with_mask(choice: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if choice is None:
        return None
    cloned = dict(choice)
    if 'mask' in cloned and isinstance(cloned['mask'], np.ndarray):
        cloned['mask'] = cloned['mask'].copy()
    return cloned


def _refresh_choice_after_mask_edit(choice: Optional[Dict[str, Any]], stats: Dict[str, Any], prompt_xy: np.ndarray) -> Optional[Dict[str, Any]]:
    """When we modify the mask (e.g., overlap trimming), refresh derived fields used in diagnostics."""
    if choice is None:
        return None
    mask = choice.get('mask', None)
    if mask is None:
        return choice
    choice = dict(choice)
    choice['area'] = _mask_area(mask)
    area_mean = float(stats.get('area_mean', 1.0) or 1.0)
    choice['area_ratio'] = float(choice['area']) / area_mean if area_mean > 1e-9 else 0.0
    choice['centroid_xy'] = _mask_centroid(mask)
    choice['prompt_inside'] = _mask_prompt_inside(mask, prompt_xy)
    return choice


def _mask_median_bgr(frame_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    pix = frame_bgr[mask]
    if pix.size == 0:
        return None
    return np.median(pix.reshape(-1, 3).astype(np.float64), axis=0)


def _point_inside_mask(mask: np.ndarray, prompt_xy: np.ndarray) -> bool:
    h, w = mask.shape[:2]
    x = int(round(float(prompt_xy[0])))
    y = int(round(float(prompt_xy[1])))
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return bool(mask[y, x])


def _default_mouse_stats(reference_bgr: np.ndarray) -> Dict[str, Any]:
    ref = np.asarray(reference_bgr, dtype=np.float64)
    return {
        'color_mean': ref.copy(),
        'color_sigma': np.array([20.0, 20.0, 20.0], dtype=np.float64),
        'area_mean': 1.0,
        'area_sigma': 1.0,
        'calibration_samples': 0,
        'reference_bgr': ref.copy(),
    }


def _build_mouse_stats(samples: List[Dict[str, Any]], reference_bgr: np.ndarray) -> Dict[str, Any]:
    if not samples:
        return _default_mouse_stats(reference_bgr)
    colors = np.stack([np.asarray(s['median_bgr'], dtype=np.float64) for s in samples], axis=0)
    areas = np.asarray([float(s['area']) for s in samples], dtype=np.float64)
    color_mean = np.median(colors, axis=0)
    color_sigma = _robust_sigma_from_mad(colors, axis=0)
    area_mean = float(np.median(areas))
    area_sigma = float(max(1.0, float(_robust_sigma_from_mad(areas))))
    return {
        'color_mean': color_mean,
        'color_sigma': color_sigma,
        'area_mean': area_mean,
        'area_sigma': area_sigma,
        'calibration_samples': int(len(samples)),
        'reference_bgr': np.asarray(reference_bgr, dtype=np.float64).copy(),
    }


def _serialize_stats(view_name: str, stats_by_mouse: Dict[int, Dict[str, Any]], render_stats: Dict[str, Any]) -> Dict[str, Any]:
    out = {'view': view_name, 'mice': {}, 'render_stats': render_stats}
    for mid, st in stats_by_mouse.items():
        out['mice'][str(mid)] = {
            'reference_bgr': np.asarray(st['reference_bgr'], dtype=np.float64).round(4).tolist(),
            'color_mean': np.asarray(st['color_mean'], dtype=np.float64).round(4).tolist(),
            'color_sigma': np.asarray(st['color_sigma'], dtype=np.float64).round(4).tolist(),
            'area_mean': float(st['area_mean']),
            'area_sigma': float(st['area_sigma']),
            'calibration_samples': int(st['calibration_samples']),
        }
    return out


def _color_distance(v: np.ndarray, ref: np.ndarray) -> float:
    d = np.asarray(v, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(d))


def _sort_prompts_by_reference(prompts_xy: np.ndarray, frame_bgr: np.ndarray, ref0: np.ndarray, ref1: np.ndarray) -> np.ndarray:
    if prompts_xy.shape[0] != 2:
        return prompts_xy
    cols = []
    for i in range(2):
        c = _median_patch(frame_bgr, float(prompts_xy[i, 0]), float(prompts_xy[i, 1]), r=5)
        if c is None:
            c = np.asarray(frame_bgr[max(0, min(frame_bgr.shape[0]-1, int(round(prompts_xy[i,1])))),
                                     max(0, min(frame_bgr.shape[1]-1, int(round(prompts_xy[i,0]))))], dtype=np.float64)
        cols.append(c)
    cost_noswap = _color_distance(cols[0], ref0) + _color_distance(cols[1], ref1)
    cost_swap = _color_distance(cols[0], ref1) + _color_distance(cols[1], ref0)
    if cost_swap < cost_noswap:
        return prompts_xy[[1, 0]].copy()
    return prompts_xy


def _sample_calibration_frames(df_cam: pd.DataFrame, max_frames: int, stride: int) -> List[int]:
    frames = sorted(df_cam['frame'].astype(int).unique().tolist())
    if not frames:
        return []
    stride = max(1, int(stride))
    sampled = frames[::stride]
    if len(sampled) > max_frames:
        idxs = np.linspace(0, len(sampled) - 1, num=max_frames, dtype=int)
        sampled = [sampled[i] for i in idxs]
    return sampled




def _prepare_frame_mapping(df_cam: pd.DataFrame, video_frame_count: int) -> Dict[str, Any]:
    """Infer how DLC frame ids correspond to video frame indices.

    Modes:
      - exact: DLC frame ids already match 0-based video frame indices
      - offset: DLC ids are consecutive with a constant offset (e.g. 1..N)
      - sequential: ignore numeric DLC ids and consume unique DLC frames in sorted order
    """
    df_cam = df_cam.sort_values('frame').reset_index(drop=True)
    unique_frames = np.asarray(sorted(pd.unique(df_cam['frame'].astype(int))), dtype=np.int64)
    if unique_frames.size == 0:
        return {'mode': 'empty', 'offset': 0, 'frame_to_pos': {}, 'unique_frames': unique_frames}

    vcount = int(max(0, video_frame_count))
    if unique_frames.min() == 0 and (vcount <= 0 or unique_frames.max() < vcount):
        return {'mode': 'exact', 'offset': 0, 'frame_to_pos': None, 'unique_frames': unique_frames}

    diffs = np.diff(unique_frames)
    consecutive = bool(unique_frames.size <= 1 or np.all(diffs == 1))
    if consecutive:
        offset = int(unique_frames[0])
        mapped_last = int(unique_frames[-1] - offset)
        if mapped_last >= 0 and (vcount <= 0 or mapped_last < vcount):
            return {'mode': 'offset', 'offset': offset, 'frame_to_pos': None, 'unique_frames': unique_frames}

    frame_to_pos = {int(f): i for i, f in enumerate(unique_frames.tolist())}
    return {'mode': 'sequential', 'offset': int(unique_frames[0]), 'frame_to_pos': frame_to_pos, 'unique_frames': unique_frames}


def _map_video_index_to_dlc_frame(video_frame_idx: int, mapping: Dict[str, Any]) -> Optional[int]:
    mode = mapping.get('mode', 'empty')
    unique_frames = mapping.get('unique_frames')
    if unique_frames is None or len(unique_frames) == 0:
        return None
    vfi = int(video_frame_idx)
    if mode == 'exact':
        return vfi
    if mode == 'offset':
        return vfi + int(mapping.get('offset', 0))
    if mode == 'sequential':
        if 0 <= vfi < len(unique_frames):
            return int(unique_frames[vfi])
        return None
    return None

def _generate_nearby_points(center_xy: np.ndarray, radius: int, mode: str) -> List[np.ndarray]:
    c = np.asarray(center_xy, dtype=np.float32)
    pts: List[np.ndarray] = [c]
    r = float(max(1, radius))
    mode = str(mode).lower()
    if mode == '8':
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dy in dirs:
            pts.append(np.array([c[0] + r*dx, c[1] + r*dy], dtype=np.float32))
    else:
        for dy in (-r, 0.0, r):
            for dx in (-r, 0.0, r):
                q = np.array([c[0] + dx, c[1] + dy], dtype=np.float32)
                if not any(np.allclose(q, x) for x in pts):
                    pts.append(q)
    return pts


def _sam_predict_candidates(predictor, point_xy: np.ndarray, multimask: bool = True) -> List[Dict[str, Any]]:
    p = np.asarray(point_xy, dtype=np.float32).reshape(1, 2)
    labels = np.array([1], dtype=np.int32)
    masks, scores, logits = predictor.predict(point_coords=p, point_labels=labels, multimask_output=bool(multimask))
    out: List[Dict[str, Any]] = []
    for i in range(len(masks)):
        mask = masks[i].astype(bool)
        out.append({
            'mask': mask,
            'sam_score': float(scores[i]),
            'prompt_xy': np.asarray(point_xy, dtype=np.float32).copy(),
            'logit_mean': float(np.mean(logits[i])) if logits is not None else 0.0,
        })
    return out


def _score_mask_candidate(frame_bgr: np.ndarray, candidate: Dict[str, Any], mouse_stats: Dict[str, Any], other_mask: Optional[np.ndarray], sam_max_factor: float, color_sigma_mult: float, expected_prompt_xy: Optional[np.ndarray] = None, prev_centroid_xy: Optional[np.ndarray] = None) -> Dict[str, Any]:
    mask = candidate['mask']
    area = _mask_area(mask)
    med = _mask_median_bgr(frame_bgr, mask)
    if med is None:
        med = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cmean = np.asarray(mouse_stats['color_mean'], dtype=np.float64)
    csig = np.maximum(np.asarray(mouse_stats['color_sigma'], dtype=np.float64), 1.0)
    area_mean = float(max(1.0, mouse_stats['area_mean']))
    area_sigma = float(max(1.0, mouse_stats['area_sigma']))

    z = np.abs(med - cmean) / csig
    color_within = bool(np.all(z <= float(color_sigma_mult)))
    area_ratio = float(area) / area_mean
    area_within = bool(area_ratio <= float(sam_max_factor))
    area_z_over = max(0.0, (float(area) - area_mean) / area_sigma)
    overlap_ratio = 0.0
    if other_mask is not None:
        inter = np.count_nonzero(mask & other_mask)
        if area > 0:
            overlap_ratio = float(inter) / float(area)
    centroid = _mask_centroid(mask)
    prompt_dist = 0.0
    if expected_prompt_xy is not None and centroid is not None:
        prompt_dist = float(np.linalg.norm(centroid - np.asarray(expected_prompt_xy, dtype=np.float64)))
    prev_dist = 0.0
    if prev_centroid_xy is not None and centroid is not None:
        prev_dist = float(np.linalg.norm(centroid - np.asarray(prev_centroid_xy, dtype=np.float64)))
    prompt_inside = _point_inside_mask(mask, np.asarray(candidate['prompt_xy'], dtype=np.float32))

    penalty = 0.0
    penalty += float(np.sum(np.maximum(0.0, z - float(color_sigma_mult)) ** 2))
    penalty += max(0.0, area_ratio - float(sam_max_factor)) ** 2 * 4.0
    penalty += overlap_ratio * 10.0
    if not prompt_inside:
        penalty += 1.5
    penalty += 0.001 * prompt_dist
    penalty += 0.0005 * prev_dist
    penalty -= 0.2 * float(candidate.get('sam_score', 0.0))

    scored = dict(candidate)
    scored.update({
        'area': area,
        'median_bgr': med,
        'color_z': z,
        'color_within': color_within,
        'area_ratio': area_ratio,
        'area_within': area_within,
        'area_z_over': area_z_over,
        'overlap_ratio': overlap_ratio,
        'centroid_xy': centroid,
        'prompt_dist': prompt_dist,
        'prev_dist': prev_dist,
        'prompt_inside': prompt_inside,
        'penalty': float(penalty),
        'passes': bool(color_within and area_within and overlap_ratio <= 0.25),
    })
    return scored


def _choose_mask_candidate(frame_bgr: np.ndarray, candidates: List[Dict[str, Any]], mouse_stats: Dict[str, Any], other_mask: Optional[np.ndarray], sam_max_factor: float, color_sigma_mult: float, expected_prompt_xy: Optional[np.ndarray], prev_centroid_xy: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    scored = [
        _score_mask_candidate(frame_bgr, c, mouse_stats, other_mask, sam_max_factor, color_sigma_mult, expected_prompt_xy, prev_centroid_xy)
        for c in candidates
    ]
    passing = [c for c in scored if c['passes']]
    pool = passing if passing else scored
    pool.sort(key=lambda c: (0 if c['passes'] else 1, c['penalty'], -float(c.get('sam_score', 0.0)), c['area_ratio']))
    return pool[0]


def _resolve_overlap_by_score(choice0: Optional[Dict[str, Any]], choice1: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if choice0 is None or choice1 is None:
        return choice0, choice1
    m0 = choice0['mask']
    m1 = choice1['mask']
    overlap = m0 & m1
    if not overlap.any():
        return choice0, choice1
    lose0 = float(choice0['penalty'])
    lose1 = float(choice1['penalty'])
    if lose0 <= lose1:
        m1 = m1 & (~overlap)
        choice1 = dict(choice1)
        choice1['mask'] = m1
        choice1['area'] = _mask_area(m1)
    else:
        m0 = m0 & (~overlap)
        choice0 = dict(choice0)
        choice0['mask'] = m0
        choice0['area'] = _mask_area(m0)
    return choice0, choice1


def _sam_segment_two_mice_v9(predictor, frame_bgr: np.ndarray, prompts_xy: np.ndarray, stats_by_mouse: Dict[int, Dict[str, Any]], sam_nearby_radius: int, sam_nearby_mode: str, sam_max_factor: float, sam_color_sigma_mult: float, prev_centroids: Optional[Dict[int, np.ndarray]] = None, debug_candidates: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    info: Dict[str, Any] = {'mouse_debug': {}, 'nearby_retry_used': {0: False, 1: False}}
    choices: Dict[int, Optional[Dict[str, Any]]] = {0: None, 1: None}

    all_candidates: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}

    def _dedup_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: List[Dict[str, Any]] = []
        seen = set()
        for c in cands:
            box = _mask_bbox(c['mask'])
            key = (box, int(np.count_nonzero(c['mask'])))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(c)
        return dedup

    def _initial_candidates(mid: int) -> List[Dict[str, Any]]:
        try:
            return _dedup_candidates(_sam_predict_candidates(predictor, prompts_xy[mid], multimask=True))
        except Exception:
            return []

    def _expand_with_nearby(mid: int, existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt_list = _generate_nearby_points(prompts_xy[mid], radius=sam_nearby_radius, mode=sam_nearby_mode)
        expanded = list(existing)
        for pxy in prompt_list:
            if float(np.linalg.norm(np.asarray(pxy, dtype=np.float64) - np.asarray(prompts_xy[mid], dtype=np.float64))) < 0.5:
                continue
            try:
                expanded.extend(_sam_predict_candidates(predictor, pxy, multimask=True))
            except Exception:
                continue
        info['nearby_retry_used'][mid] = True
        return _dedup_candidates(expanded)

    for mid in (0, 1):
        all_candidates[mid] = _initial_candidates(mid)

    # Fast path: center prompt only. Nearby retry is used only if the chosen mask fails checks or nothing was found.
    choice0 = _choose_mask_candidate(frame_bgr, all_candidates[0], stats_by_mouse[0], None, sam_max_factor, sam_color_sigma_mult, prompts_xy[0], None if prev_centroids is None else prev_centroids.get(0))
    if choice0 is None or (not bool(choice0.get('passes', False))):
        all_candidates[0] = _expand_with_nearby(0, all_candidates[0])
        choice0 = _choose_mask_candidate(frame_bgr, all_candidates[0], stats_by_mouse[0], None, sam_max_factor, sam_color_sigma_mult, prompts_xy[0], None if prev_centroids is None else prev_centroids.get(0))

    other0 = None if choice0 is None else choice0['mask']
    choice1 = _choose_mask_candidate(frame_bgr, all_candidates[1], stats_by_mouse[1], other0, sam_max_factor, sam_color_sigma_mult, prompts_xy[1], None if prev_centroids is None else prev_centroids.get(1))
    if choice1 is None or (not bool(choice1.get('passes', False))):
        all_candidates[1] = _expand_with_nearby(1, all_candidates[1])
        choice1 = _choose_mask_candidate(frame_bgr, all_candidates[1], stats_by_mouse[1], other0, sam_max_factor, sam_color_sigma_mult, prompts_xy[1], None if prev_centroids is None else prev_centroids.get(1))

    # Re-check mouse 0 with overlap information from mouse 1. Retry nearby only if overlap-aware choice still fails.
    other1 = None if choice1 is None else choice1['mask']
    choice0b = _choose_mask_candidate(frame_bgr, all_candidates[0], stats_by_mouse[0], other1, sam_max_factor, sam_color_sigma_mult, prompts_xy[0], None if prev_centroids is None else prev_centroids.get(0))
    if choice0b is not None:
        choice0 = choice0b
    if choice0 is None or (not bool(choice0.get('passes', False))):
        if not info['nearby_retry_used'][0]:
            all_candidates[0] = _expand_with_nearby(0, all_candidates[0])
        choice0b = _choose_mask_candidate(frame_bgr, all_candidates[0], stats_by_mouse[0], other1, sam_max_factor, sam_color_sigma_mult, prompts_xy[0], None if prev_centroids is None else prev_centroids.get(0))
        if choice0b is not None:
            choice0 = choice0b

    pre_overlap = {0: _clone_choice_with_mask(choice0), 1: _clone_choice_with_mask(choice1)}

    choice0, choice1 = _resolve_overlap_by_score(choice0, choice1)
    choice0 = _refresh_choice_after_mask_edit(choice0, stats_by_mouse[0], prompts_xy[0])
    choice1 = _refresh_choice_after_mask_edit(choice1, stats_by_mouse[1], prompts_xy[1])
    post_overlap = {0: _clone_choice_with_mask(choice0), 1: _clone_choice_with_mask(choice1)}
    choices[0], choices[1] = choice0, choice1

    for mid in (0, 1):
        ch = choices[mid]
        if ch is None or ch.get('area', 0) <= 0:
            if ch is not None:
                ch['status'] = 'no_mask'
            choices[mid] = None if ch is None or ch.get('area', 0) <= 0 else ch
            continue
        if bool(ch.get('passes', False)):
            ch['status'] = 'nearby_retry_accept' if bool(info['nearby_retry_used'][mid]) else 'direct_accept'
        else:
            ch['status'] = 'fallback_accept'

    for mid in (0, 1):
        ch = choices[mid]
        final_prompt_inside = None if ch is None else _mask_prompt_inside(ch['mask'], prompts_xy[mid])
        info['mouse_debug'][mid] = {
            'num_candidates': len(all_candidates[mid]),
            'nearby_retry_used': bool(info['nearby_retry_used'][mid]),
            'selected_passes': None if ch is None else bool(ch['passes']),
            'selected_penalty': None if ch is None else float(ch['penalty']),
            'selected_area': None if ch is None else int(ch['area']),
            'selected_bgr': None if ch is None else np.asarray(ch['median_bgr']).round(3).tolist(),
    'selected_status': 'no_mask' if ch is None else str(ch.get('status', 'fallback_accept')),
    'best_pre_overlap_candidate': _choice_log_entry(pre_overlap.get(mid), prompts_xy[mid], stats_by_mouse[mid], include_mask=debug_candidates),
    'final_post_overlap_candidate': _choice_log_entry(post_overlap.get(mid), prompts_xy[mid], stats_by_mouse[mid], include_mask=debug_candidates),
    'final_displayed_area': 0 if ch is None else int(ch.get('area', 0) or 0),
    'final_displayed_status': 'no_mask' if ch is None else str(ch.get('status', 'fallback_accept')),
    'final_prompt_inside': final_prompt_inside,
    'final_prompt_inside_false': None if final_prompt_inside is None else (not bool(final_prompt_inside)),
}
        if debug_candidates:
            info['mouse_debug'][mid]['top_candidates'] = [
                {
                    'passes': bool(c['passes']),
                    'penalty': float(c['penalty']),
                    'sam_score': float(c.get('sam_score', 0.0)),
                    'area': int(c['area']),
                    'area_ratio': float(c['area_ratio']),
                    'color_z': np.asarray(c['color_z']).round(3).tolist(),
                    'prompt_inside': bool(c['prompt_inside']),
                }
                for c in sorted([_score_mask_candidate(frame_bgr, cand, stats_by_mouse[mid], None, sam_max_factor, sam_color_sigma_mult, prompts_xy[mid], None if prev_centroids is None else prev_centroids.get(mid)) for cand in all_candidates[mid]], key=lambda x: (0 if x['passes'] else 1, x['penalty']))[:8]
            ]
    return choices[0], choices[1], info

def _overlay_masks_constant_color(frame_bgr: np.ndarray, mask0: Optional[np.ndarray], mask1: Optional[np.ndarray], color0_bgr: np.ndarray, color1_bgr: np.ndarray, alpha: float) -> np.ndarray:
    out = frame_bgr.astype(np.float32)
    overlay_colors = [np.asarray(color0_bgr, dtype=np.float32), np.asarray(color1_bgr, dtype=np.float32)]
    for mask, col in zip((mask0, mask1), overlay_colors):
        if mask is None or mask.sum() == 0:
            continue
        out[mask] = (1.0 - float(alpha)) * out[mask] + float(alpha) * col.reshape(1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)



def _sam_segment_two_mice_v5style(predictor, frame_bgr: np.ndarray, prompts_xy: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    out: Dict[int, Optional[Dict[str, Any]]] = {0: None, 1: None}
    debug: Dict[str, Any] = {'mouse_debug': {}, 'nearby_retry_used': {0: False, 1: False}}
    for mid in (0, 1):
        try:
            cands = _sam_predict_candidates(predictor, prompts_xy[mid], multimask=True)
        except Exception:
            cands = []
        if cands:
            best = max(cands, key=lambda c: float(c.get('sam_score', 0.0)))
            mask = best['mask']
            med = _mask_median_bgr(frame_bgr, mask)
            if med is None:
                med = np.zeros(3, dtype=np.float64)
            out[mid] = {
                'mask': mask,
                'sam_score': float(best.get('sam_score', 0.0)),
                'prompt_xy': np.asarray(prompts_xy[mid], dtype=np.float32).copy(),
                'area': _mask_area(mask),
                'median_bgr': np.asarray(med, dtype=np.float64),
                'centroid_xy': _mask_centroid(mask),
                'passes': True,
                'status': 'direct_accept',
                'penalty': 0.0,
            }
        debug['mouse_debug'][mid] = {
            'num_candidates': len(cands),
            'nearby_retry_used': False,
            'selected_passes': bool(out[mid] is not None),
            'selected_penalty': None if out[mid] is None else 0.0,
            'selected_area': None if out[mid] is None else int(out[mid]['area']),
            'selected_bgr': None if out[mid] is None else np.asarray(out[mid]['median_bgr']).round(3).tolist(),
            'selected_status': 'no_mask' if out[mid] is None else 'direct_accept',
        }
    return out[0], out[1], debug


def _status_priority(status: str) -> int:
    order = {
        'direct_accept': 0,
        'nearby_retry_accept': 1,
        'fallback_accept': 2,
        'no_mask': 3,
    }
    return order.get(str(status), 3)


def _frame_status_from_mouse_statuses(statuses: List[str]) -> str:
    if not statuses:
        return 'no_mask'
    return max(statuses, key=_status_priority)


def _draw_status_bulbs(frame_bgr: np.ndarray, status_mouse0: str, status_mouse1: str) -> np.ndarray:
    colors = {
        'direct_accept': (0, 220, 0),
        'nearby_retry_accept': (0, 255, 255),
        'fallback_accept': (0, 165, 255),
        'no_mask': (0, 0, 255),
    }
    out = frame_bgr
    h, w = out.shape[:2]
    r = 10
    centers = [
        (22, 22),
        (max(14, w - 22), 22),
    ]
    statuses = [status_mouse0, status_mouse1]
    for center, status in zip(centers, statuses):
        cv2.circle(out, center, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, center, r, colors.get(status, (0, 0, 255)), -1, cv2.LINE_AA)
    return out


def _draw_frame_number(frame_bgr: np.ndarray, frame_number: int) -> np.ndarray:
    out = frame_bgr
    h, w = out.shape[:2]
    text = str(int(frame_number))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness_fg = 1
    thickness_bg = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness_fg)
    org = (max(4, (w - tw) // 2), max(18, 22 + th // 2))
    cv2.putText(out, text, org, font, scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
    cv2.putText(out, text, org, font, scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
    return out


def _collect_calibration_samples(predictor, frame_bgr: np.ndarray, rows: pd.DataFrame, sam_min_lik: float, sam_use_dlc_ids: bool, manual0: np.ndarray, manual1: np.ndarray, sam_nearby_radius: int, sam_nearby_mode: str, sam_max_factor: float, sam_color_sigma_mult: float) -> Optional[Dict[int, Dict[str, Any]]]:
    prompts = _two_mouse_prompts_from_rows(rows, min_lik=sam_min_lik, use_dlc_ids=bool(sam_use_dlc_ids))
    if prompts is None:
        return None
    prompts = _sort_prompts_by_reference(prompts, frame_bgr, manual0, manual1)
    default_stats = {0: _default_mouse_stats(manual0), 1: _default_mouse_stats(manual1)}
    ch0, ch1, _ = _sam_segment_two_mice_v9(predictor, frame_bgr, prompts, default_stats, sam_nearby_radius, sam_nearby_mode, sam_max_factor=max(2.5, sam_max_factor), sam_color_sigma_mult=max(4.0, sam_color_sigma_mult))
    if ch0 is None or ch1 is None:
        return None
    if ch0['area'] <= 0 or ch1['area'] <= 0:
        return None
    if (ch0['mask'] & ch1['mask']).any():
        return None
    return {0: {'median_bgr': ch0['median_bgr'], 'area': ch0['area']}, 1: {'median_bgr': ch1['median_bgr'], 'area': ch1['area']}}


def _learn_sam_stats_for_video(predictor, video_path: str | Path, df_cam: pd.DataFrame, manual0: np.ndarray, manual1: np.ndarray, sam_min_lik: float, sam_use_dlc_ids: bool, sam_calib_frames: int, sam_calib_stride: int, sam_nearby_radius: int, sam_nearby_mode: str, sam_max_factor: float, sam_color_sigma_mult: float) -> Dict[int, Dict[str, Any]]:
    stats_samples: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open input video for SAM calibration: {video_path}')

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    df_cam = df_cam.sort_values('frame').reset_index(drop=True)
    mapping = _prepare_frame_mapping(df_cam, video_frame_count)
    unique_frames = mapping.get('unique_frames', np.asarray([], dtype=np.int64))
    sampled_dlc_frames = _sample_calibration_frames(df_cam, sam_calib_frames, sam_calib_stride)
    target_dlc_frames = set(int(x) for x in sampled_dlc_frames)
    if not target_dlc_frames:
        cap.release()
        return {0: _default_mouse_stats(manual0), 1: _default_mouse_stats(manual1)}

    frames_arr = df_cam['frame'].to_numpy(dtype=np.int64)
    nrows = len(df_cam)
    idx = 0
    fr = 0
    max_video_fr = len(unique_frames) - 1 if mapping.get('mode') == 'sequential' else max(0, int(unique_frames.max() - int(mapping.get('offset', 0))))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fr > max_video_fr:
            break
        dlc_frame = _map_video_index_to_dlc_frame(fr, mapping)
        if dlc_frame is None:
            break
        while idx < nrows and frames_arr[idx] < dlc_frame:
            idx += 1
        start = idx
        while idx < nrows and frames_arr[idx] == dlc_frame:
            idx += 1
        if dlc_frame in target_dlc_frames and idx > start:
            rows = df_cam.iloc[start:idx]
            sample = _collect_calibration_samples(predictor, frame, rows, sam_min_lik, sam_use_dlc_ids, manual0, manual1, sam_nearby_radius, sam_nearby_mode, sam_max_factor, sam_color_sigma_mult)
            if sample is not None:
                for mid in (0, 1):
                    stats_samples[mid].append(sample[mid])
        fr += 1
    cap.release()
    return {0: _build_mouse_stats(stats_samples[0], manual0), 1: _build_mouse_stats(stats_samples[1], manual1)}


def overlay_video_with_sam(
    in_path: str | Path,
    out_path: str | Path,
    df_cam: pd.DataFrame,
    edges: Optional[List[Tuple[str, str]]],
    likelihood_min_draw: float,
    draw_labels: bool,
    label_scale: float,
    label_thickness: int,
    force_fps: Optional[float],
    sam_checkpoint: str,
    sam_model_type: str,
    sam_device: str,
    sam_alpha: float,
    sam_min_lik: float,
    sam_use_dlc_ids: bool,
    sam_every: int,
    sam_only: bool,
    mouse0_bgr: np.ndarray,
    mouse1_bgr: np.ndarray,
    sam_calib_frames: int,
    sam_calib_stride: int,
    sam_max_factor: float,
    sam_color_sigma_mult: float,
    sam_nearby_radius: int,
    sam_nearby_mode: str,
    sam_debug_candidates: bool,
    sam_stats_out: Optional[str],
    view_name: str,
    sam_selection_mode: str = 'robust',
    number_of_frames: int = 0,
    frame_interval: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    predictor = _sam_load_predictor(sam_checkpoint, sam_model_type, sam_device)
    stats_by_mouse = _learn_sam_stats_for_video(
        predictor, in_path, df_cam, mouse0_bgr, mouse1_bgr, sam_min_lik, sam_use_dlc_ids,
        sam_calib_frames, sam_calib_stride, sam_nearby_radius, sam_nearby_mode, sam_max_factor, sam_color_sigma_mult
    )
    print(f"[SAM:{view_name}] Calibration mouse0 color~{np.asarray(stats_by_mouse[0]['color_mean']).round(1).tolist()} sigma~{np.asarray(stats_by_mouse[0]['color_sigma']).round(1).tolist()} area~{stats_by_mouse[0]['area_mean']:.1f}")
    print(f"[SAM:{view_name}] Calibration mouse1 color~{np.asarray(stats_by_mouse[1]['color_mean']).round(1).tolist()} sigma~{np.asarray(stats_by_mouse[1]['color_sigma']).round(1).tolist()} area~{stats_by_mouse[1]['area_mean']:.1f}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    fps = float(force_fps) if force_fps is not None else (fps_in if fps_in > 1e-6 else 10.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    df_cam = df_cam.sort_values('frame').reset_index(drop=True)
    mapping = _prepare_frame_mapping(df_cam, video_frame_count)
    unique_frames = mapping.get('unique_frames', np.asarray([], dtype=np.int64))
    print(f"[SAM:{view_name}] frame mapping mode={mapping.get('mode')} offset={mapping.get('offset', 0)} unique_dlc_frames={len(unique_frames)} video_frames={video_frame_count}")
    frames_arr = df_cam['frame'].to_numpy(dtype=np.int64)
    nrows = len(df_cam)
    idx = 0
    last_prompts = None
    prev_centroids: Dict[int, np.ndarray] = {}
    # Keep logs compact for long runs: store aggregate counters (and a small capped list
    # of frame indices) instead of embedding large per-frame debug payloads.
    _MAX_FLAG_FRAMES_PER_MOUSE = 25
    render_stats: Dict[str, Any] = {
        'frames_total': 0, 'frames_with_prompts': 0, 'frames_segmented': 0,
        'mouse0_selected_pass': 0, 'mouse1_selected_pass': 0,
        'mouse0_selected_fail': 0, 'mouse1_selected_fail': 0,
        'mouse0_nearby_retry_used': 0, 'mouse1_nearby_retry_used': 0,
        'frame_status_counts': {'direct_accept': 0, 'nearby_retry_accept': 0, 'fallback_accept': 0, 'no_mask': 0},
        'mouse_status_counts': {
            '0': {'direct_accept': 0, 'nearby_retry_accept': 0, 'fallback_accept': 0, 'no_mask': 0},
            '1': {'direct_accept': 0, 'nearby_retry_accept': 0, 'fallback_accept': 0, 'no_mask': 0},
        },
        # Aggregate diagnostics for the *final displayed* masks (post-overlap).
        'final_prompt_inside_false_count': {'0': 0, '1': 0},
        'final_prompt_inside_false_frames': {'0': [], '1': []},
        'final_displayed_area_min': {'0': None, '1': None},
        'final_displayed_area_min_frame': {'0': None, '1': None},
        'error_count': 0,
        'error_frames': [],
        'selection_mode': str(sam_selection_mode),
        # Keep key for backward compatibility, but do not populate it for long runs.
        'debug_frames': []
    }
    samples_running = {0: {'colors': [], 'areas': []}, 1: {'colors': [], 'areas': []}}
    max_frames_to_process = max(0, int(number_of_frames))
    if max_frames_to_process > 0:
        print(f"[SAM:{view_name}] Production pass limited to {max_frames_to_process} frame(s) by --number-of-frames={max_frames_to_process}")
    interval_begin = 0
    interval_end = max(0, video_frame_count - 1)
    if frame_interval is not None:
        interval_begin = max(0, int(frame_interval[0]))
        interval_end = max(interval_begin, int(frame_interval[1]))
        print(f"[SAM:{view_name}] Production frame interval limited to [{interval_begin}, {interval_end}] by --frame-interval={interval_begin},{interval_end}")

    source_frame_idx = 0
    processed_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source_frame_idx < interval_begin:
            source_frame_idx += 1
            continue
        if source_frame_idx > interval_end:
            break
        if max_frames_to_process > 0 and processed_frames >= max_frames_to_process:
            break
        render_stats['frames_total'] += 1
        dlc_frame = _map_video_index_to_dlc_frame(source_frame_idx, mapping)
        if dlc_frame is None:
            rows = df_cam.iloc[0:0]
        else:
            while idx < nrows and frames_arr[idx] < dlc_frame:
                idx += 1
            start = idx
            while idx < nrows and frames_arr[idx] == dlc_frame:
                idx += 1
            rows = df_cam.iloc[start:idx] if idx > start else df_cam.iloc[0:0]

        out_frame = frame
        frame_status = 'no_mask'
        mouse_statuses: Dict[int, str] = {0: 'no_mask', 1: 'no_mask'}
        if sam_every <= 1 or (source_frame_idx % sam_every == 0):
            prompts = _two_mouse_prompts_from_rows(rows, min_lik=sam_min_lik, use_dlc_ids=bool(sam_use_dlc_ids))
            if prompts is None:
                prompts = last_prompts
            else:
                prompts = _sort_prompts_by_reference(prompts, frame, mouse0_bgr, mouse1_bgr)
                last_prompts = prompts
            if prompts is not None:
                render_stats['frames_with_prompts'] += 1
                try:
                    if str(sam_selection_mode) == 'v5':
                        ch0, ch1, debug = _sam_segment_two_mice_v5style(predictor, out_frame, prompts)
                    else:
                        ch0, ch1, debug = _sam_segment_two_mice_v9(
                            predictor, out_frame, prompts, stats_by_mouse, sam_nearby_radius, sam_nearby_mode,
                            sam_max_factor, sam_color_sigma_mult, prev_centroids=prev_centroids, debug_candidates=sam_debug_candidates
                        )
                    m0 = None if ch0 is None else ch0['mask']
                    m1 = None if ch1 is None else ch1['mask']
                    for mid, ch in ((0, ch0), (1, ch1)):
                        status = 'no_mask' if ch is None else str(ch.get('status', 'fallback_accept'))
                        mouse_statuses[mid] = status
                        render_stats['mouse_status_counts'][str(mid)][status] += 1
                        if bool(debug.get('nearby_retry_used', {}).get(mid, False)):
                            render_stats[f'mouse{mid}_nearby_retry_used'] += 1
                        if ch is not None and ch.get('centroid_xy') is not None:
                            prev_centroids[mid] = np.asarray(ch['centroid_xy'], dtype=np.float64)
                            samples_running[mid]['colors'].append(np.asarray(ch['median_bgr'], dtype=np.float64))
                            samples_running[mid]['areas'].append(float(ch['area']))
                        if ch is not None and bool(ch.get('passes', False)):
                            render_stats[f'mouse{mid}_selected_pass'] += 1
                        else:
                            render_stats[f'mouse{mid}_selected_fail'] += 1

                        # ---- Aggregate diagnostics (compact; no masks stored) ----
                        displayed_area = 0 if ch is None else int(ch.get('area', 0) or 0)
                        if displayed_area > 0:
                            cur_min = render_stats['final_displayed_area_min'][str(mid)]
                            if cur_min is None or displayed_area < int(cur_min):
                                render_stats['final_displayed_area_min'][str(mid)] = int(displayed_area)
                                render_stats['final_displayed_area_min_frame'][str(mid)] = int(source_frame_idx)

                        # Prefer post-overlap prompt-inside flag from the segmenter (v9).
                        final_pi_false = None
                        try:
                            final_pi_false = debug.get('mouse_debug', {}).get(mid, {}).get('final_prompt_inside_false', None)
                        except Exception:
                            final_pi_false = None
                        if final_pi_false is None and ch is not None:
                            # Fallback computation for modes that don't populate it.
                            try:
                                final_pi_false = (not bool(_mask_prompt_inside(ch['mask'], prompts[mid])))
                            except Exception:
                                final_pi_false = None
                        if final_pi_false is True:
                            render_stats['final_prompt_inside_false_count'][str(mid)] += 1
                            lst = render_stats['final_prompt_inside_false_frames'][str(mid)]
                            if len(lst) < _MAX_FLAG_FRAMES_PER_MOUSE:
                                lst.append(int(source_frame_idx))
                    frame_status = _frame_status_from_mouse_statuses([mouse_statuses[0], mouse_statuses[1]])
                    render_stats['frame_status_counts'][frame_status] += 1
                    if m0 is not None or m1 is not None:
                        overlay0 = np.asarray(stats_by_mouse[0].get('color_mean', mouse0_bgr), dtype=np.float32)
                        overlay1 = np.asarray(stats_by_mouse[1].get('color_mean', mouse1_bgr), dtype=np.float32)
                        out_frame = _overlay_masks_constant_color(out_frame, m0, m1, overlay0, overlay1, alpha=float(sam_alpha))
                        render_stats['frames_segmented'] += 1
                    cv2.circle(out_frame, (int(prompts[0, 0]), int(prompts[0, 1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(out_frame, (int(prompts[1, 0]), int(prompts[1, 1])), 4, (80, 220, 80), -1, cv2.LINE_AA)
                except Exception as exc:
                    frame_status = 'no_mask'
                    render_stats['frame_status_counts'][frame_status] += 1
                    render_stats['mouse_status_counts']['0']['no_mask'] += 1
                    render_stats['mouse_status_counts']['1']['no_mask'] += 1
                    render_stats['error_count'] += 1
                    if len(render_stats['error_frames']) < 25:
                        render_stats['error_frames'].append({'frame': int(source_frame_idx), 'error': str(exc)})
            else:
                render_stats['frame_status_counts'][frame_status] += 1
                render_stats['mouse_status_counts']['0']['no_mask'] += 1
                render_stats['mouse_status_counts']['1']['no_mask'] += 1
        else:
            render_stats['frame_status_counts'][frame_status] += 1
            render_stats['mouse_status_counts']['0']['no_mask'] += 1
            render_stats['mouse_status_counts']['1']['no_mask'] += 1
        out_frame = _draw_status_bulbs(out_frame, mouse_statuses[0], mouse_statuses[1])
        out_frame = _draw_frame_number(out_frame, source_frame_idx)
        if not sam_only:
            out_frame = overlay_frame(out_frame, rows, edges, likelihood_min_draw, draw_labels, label_scale, label_thickness)
        writer.write(out_frame)
        processed_frames += 1
        source_frame_idx += 1

    cap.release()
    writer.release()

    for mid in (0, 1):
        colors = samples_running[mid]['colors']
        areas = samples_running[mid]['areas']
        if colors:
            carr = np.stack(colors, axis=0)
            render_stats[f'mouse{mid}_mean_color'] = np.mean(carr, axis=0).round(4).tolist()
            render_stats[f'mouse{mid}_std_color'] = np.std(carr, axis=0, ddof=0).round(4).tolist()
        else:
            render_stats[f'mouse{mid}_mean_color'] = None
            render_stats[f'mouse{mid}_std_color'] = None
        if areas:
            aarr = np.asarray(areas, dtype=np.float64)
            render_stats[f'mouse{mid}_mean_area'] = float(np.mean(aarr))
            render_stats[f'mouse{mid}_std_area'] = float(np.std(aarr, ddof=0))
        else:
            render_stats[f'mouse{mid}_mean_area'] = None
            render_stats[f'mouse{mid}_std_area'] = None

    payload = _serialize_stats(view_name, stats_by_mouse, render_stats)
    if sam_stats_out:
        stats_path = Path(sam_stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"[SAM:{view_name}] Wrote stats: {stats_path}")
    return payload


def join_quadrants_from_videos(
    top_video: str | Path,
    front_video: str | Path,
    side_video: str | Path,
    joined_out: str | Path,
    force_fps: Optional[float],
) -> None:
    # Create joined mosaic from already-rendered videos using INTER_AREA downsampling.
    cap_t = cv2.VideoCapture(str(top_video))
    cap_f = cv2.VideoCapture(str(front_video))
    cap_s = cv2.VideoCapture(str(side_video))
    if not cap_t.isOpened() or not cap_f.isOpened() or not cap_s.isOpened():
        raise RuntimeError("Could not open one of the annotated videos for joining.")

    w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap_t.get(cv2.CAP_PROP_FPS))
    fps = float(force_fps) if force_fps is not None else (fps_in if fps_in > 1e-6 else 10.0)

    w_left = w // 2
    w_right = w - w_left
    h_top = h // 2
    h_bottom = h - h_top

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(joined_out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open joined output video: {joined_out}")

    while True:
        ok_t, ft = cap_t.read()
        ok_f, ff = cap_f.read()
        ok_s, fs = cap_s.read()
        if not (ok_t and ok_f and ok_s):
            break

        if ff.shape[1] != w or ff.shape[0] != h:
            ff = cv2.resize(ff, (w, h), interpolation=cv2.INTER_LINEAR)
        if fs.shape[1] != w or fs.shape[0] != h:
            fs = cv2.resize(fs, (w, h), interpolation=cv2.INTER_LINEAR)

        top_small = cv2.resize(ft, (w_right, h_top), interpolation=cv2.INTER_AREA)
        front_small = cv2.resize(ff, (w_right, h_bottom), interpolation=cv2.INTER_AREA)
        side_small = cv2.resize(fs, (w_left, h_bottom), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[0:h_top, w_left:w] = top_small
        canvas[h_top:h, 0:w_left] = side_small
        canvas[h_top:h, w_left:w] = front_small
        writer.write(canvas)

    cap_t.release()
    cap_f.release()
    cap_s.release()
    writer.release()

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="overlay_dlc_on_videos_v2.py")
    p.add_argument("--dlc", required=True, help="Path to dlc_long.csv")

    p.add_argument("--topIn", required=True, help="Input top camera video")
    p.add_argument("--frontIn", required=True, help="Input front camera video")
    p.add_argument("--sideIn", required=True, help="Input side camera video")

    p.add_argument("--topOut", required=True, help="Output annotated top video")
    p.add_argument("--frontOut", required=True, help="Output annotated front video")
    p.add_argument("--sideOut", required=True, help="Output annotated side video")

    p.add_argument("--joinedVideo", default=None,
                   help="Optional output path for a joined mosaic video (same resolution as inputs).")

    p.add_argument("--topCam", default="cam1_top", help="Camera name in CSV for top view (default cam1_top)")
    p.add_argument("--frontCam", default="cam2_front", help="Camera name in CSV for front view (default cam2_front)")
    p.add_argument("--sideCam", default="cam3_side", help="Camera name in CSV for side view (default cam3_side)")

    p.add_argument("--template", default=None, help="Optional template file to draw skeleton edges ([EDGES] section)")

    p.add_argument("--likelihood-min", type=float, default=0.2, help="Only draw points with likelihood >= this (default 0.2)")
    p.add_argument("--draw-labels", action="store_true", help="Draw text labels 'mouse:bodypart' near each point")
    p.add_argument("--label-scale", type=float, default=0.4, help="OpenCV font scale for labels (default 0.4)")
    p.add_argument("--label-thickness", type=int, default=1, help="Label thickness (default 1)")
    p.add_argument("--force-fps", type=float, default=None, help="Override output FPS (else use input video FPS)")

    # SAM segmentation visualization
    p.add_argument("--sam", action="store_true", help="Overlay Segment Anything (SAM) two-mouse segments (visualization/debug).")
    p.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint .pth (required if --sam).")
    p.add_argument("--sam-model-type", default="vit_b", help="SAM model type: vit_b, vit_l, vit_h (default vit_b).")
    p.add_argument("--sam-device", default="auto", help="Device for SAM: auto, cpu, cuda (default auto).")
    p.add_argument("--sam-alpha", type=float, default=0.65, help="Alpha blend for constant-color masks (default 0.65).")
    p.add_argument("--sam-min-likelihood", type=float, default=0.5, help="Min DLC likelihood for centroid prompts (default 0.5).")
    p.add_argument("--sam-use-dlc-ids", action="store_true", help="Use DLC individual IDs to compute prompts (else XY kmeans).")
    p.add_argument("--sam-every", type=int, default=1, help="Run SAM every N frames (default 1).")
    p.add_argument("--sam-only", action="store_true", help="Render only SAM segments (no skeleton/points).")
    p.add_argument("--sam-calib-frames", type=int, default=60, help="Number of frames sampled for SAM calibration statistics (default 60).")
    p.add_argument("--sam-calib-stride", type=int, default=0, help="Calibration stride. 0 means automatic even spacing across the full video (default 0).")
    p.add_argument("--sam-max-factor", type=float, default=2.0, help="Maximum allowed segment area relative to learned median area (default 2.0).")
    p.add_argument("--sam-color-sigma-mult", type=float, default=3.0, help="Allowed color deviation in robust sigmas per BGR channel (default 3.0).")
    p.add_argument("--sam-nearby-radius", type=int, default=12, help="Pixel radius used when retrying SAM prompts near the original point (default 12).")
    p.add_argument("--sam-nearby-mode", choices=["8", "grid"], default="grid", help="Retry pattern for nearby SAM prompts: 8 directions or 3x3 grid (default grid).")
    p.add_argument("--sam-debug-candidates", action="store_true", help="Record extra candidate-ranking diagnostics in the SAM stats JSON.")
    p.add_argument("--sam-stats-out", default=None, help="Optional JSON path prefix for SAM statistics. Per-view suffixes _top/_front/_side are added.")
    p.add_argument("--sam-selection-mode", choices=["robust", "v5"], default="robust", help="SAM candidate selection mode: robust or v5 (default robust).")
    p.add_argument("--number-of-frames", type=int, default=0, help="Limit production rendering to the first N processed frames. 0 means full video (default 0).")
    p.add_argument("--frame-interval", default=None, help="Limit production rendering to an inclusive source-frame interval begin,end (for example 500,650).")


    # Identity correction (2 mice, color-based)
    p.add_argument("--autoCorrect", action="store_true",
                   help="Enable color-based identity correction (2 mice).")
    p.add_argument("--dlcOut", default=None,
                   help="Optional output path for corrected dlc CSV.")
    p.add_argument("--sep-likelihood", type=float, default=0.7,
                   help="Likelihood threshold used to find separated frames (default 0.7).")
    p.add_argument("--calib-frames", type=int, default=25,
                   help="Number of separated frames used for color calibration (default 25).")
    p.add_argument("--calib-likelihood", type=float, default=0.85,
                   help="Likelihood threshold for using detections in calibration (default 0.85).")
    p.add_argument("--vote-likelihood", type=float, default=0.5,
                   help="Likelihood threshold for color voting/correction (default 0.5).")
    p.add_argument("--patch-radius", type=int, default=6,
                   help="Patch radius in pixels around each detection (default 6 => 13x13).")
    p.add_argument("--color-dist", type=float, default=35.0,
                   help="Color distance threshold in BGR for counting fur pixels (default 35).")
    p.add_argument("--min-fur-frac", type=float, default=0.08,
                   help="Minimum fraction of patch pixels close to a prototype (default 0.08).")
    p.add_argument("--min-pixels", type=int, default=30,
                   help="Minimum number of patch pixels required (default 30).")
    p.add_argument("--dominance", type=float, default=0.25,
                   help="Dominance margin for best-vs-second cluster counts (default 0.25).")
    p.add_argument("--drop-on-conflict", action="store_true",
                   help="If a part wants to switch but the target already has it, drop the leaving part.")

    p.add_argument("--mouse0-bgr", default=None,
                   help="Manual BGR prototype for mouse0, format: 'b,g,r'. Use with --mouse1-bgr.")
    p.add_argument("--mouse1-bgr", default=None,
                   help="Manual BGR prototype for mouse1, format: 'b,g,r'. Use with --mouse0-bgr.")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    frame_interval = None
    if args.frame_interval is not None:
        try:
            a, b = [int(x.strip()) for x in str(args.frame_interval).split(',', 1)]
        except Exception as exc:
            raise ValueError('--frame-interval must have the form beginFrameID,endFrameID') from exc
        if a < 0 or b < 0:
            raise ValueError('--frame-interval requires non-negative frame indices')
        frame_interval = (min(a, b), max(a, b))

    df = pd.read_csv(args.dlc)
    required = {"frame", "camera", "individual", "bodypart", "x", "y", "likelihood"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dlc CSV missing columns: {sorted(missing)}")

    edges = None
    if args.template:
        edges = parse_edges_from_template(args.template)
        if not edges:
            print("[warn] Template provided but no edges found; drawing only points.")

    def df_for_cam(cam_name: str) -> pd.DataFrame:
        d = df[df["camera"].astype(str) == str(cam_name)].copy()
        d["frame"] = d["frame"].astype(int)
        return d

    print("Loading DLC rows...")
    df_top = df_for_cam(args.topCam)
    df_front = df_for_cam(args.frontCam)
    df_side = df_for_cam(args.sideCam)
    print(f"Rows per view: top={len(df_top)}, front={len(df_front)}, side={len(df_side)}")
    manual0 = manual1 = None
    if args.mouse0_bgr is not None or args.mouse1_bgr is not None:
        if args.mouse0_bgr is None or args.mouse1_bgr is None:
            raise ValueError("Provide both --mouse0-bgr and --mouse1-bgr to use manual prototypes.")
        manual0 = parse_bgr_triplet(args.mouse0_bgr)
        manual1 = parse_bgr_triplet(args.mouse1_bgr)
    if args.sam and (manual0 is None or manual1 is None):
        raise ValueError("SAM v9 requires --mouse0-bgr and --mouse1-bgr for statistical segment assignment.")

    def sam_stats_path_for(view_name: str) -> Optional[str]:
        if not args.sam_stats_out:
            return None
        p = Path(args.sam_stats_out)
        suffix = p.suffix if p.suffix else '.json'
        stem = p.stem if p.suffix else p.name
        parent = p.parent if p.parent != Path('') else Path('.')
        return str(parent / f"{stem}_{view_name}{suffix}")

    if args.autoCorrect:
        print("[autoCorrect] Top view...")
        df_top = correct_identities_for_camera(
            df_top, args.topIn,
            sep_likelihood=float(args.sep_likelihood),
            calib_frames=int(args.calib_frames),
            calib_likelihood=float(args.calib_likelihood),
            vote_likelihood=float(args.vote_likelihood),
            patch_radius=int(args.patch_radius),
            color_dist=float(args.color_dist),
            min_fur_frac=float(args.min_fur_frac),
            min_pixels=int(args.min_pixels),
            dominance=float(args.dominance),
            drop_on_conflict=bool(args.drop_on_conflict),
            manual_mouse0_bgr=manual0,
            manual_mouse1_bgr=manual1,
        )
        print("[autoCorrect] Front view...")
        df_front = correct_identities_for_camera(
            df_front, args.frontIn,
            sep_likelihood=float(args.sep_likelihood),
            calib_frames=int(args.calib_frames),
            calib_likelihood=float(args.calib_likelihood),
            vote_likelihood=float(args.vote_likelihood),
            patch_radius=int(args.patch_radius),
            color_dist=float(args.color_dist),
            min_fur_frac=float(args.min_fur_frac),
            min_pixels=int(args.min_pixels),
            dominance=float(args.dominance),
            drop_on_conflict=bool(args.drop_on_conflict),
            manual_mouse0_bgr=manual0,
            manual_mouse1_bgr=manual1,
        )
        print("[autoCorrect] Side view...")
        df_side = correct_identities_for_camera(
            df_side, args.sideIn,
            sep_likelihood=float(args.sep_likelihood),
            calib_frames=int(args.calib_frames),
            calib_likelihood=float(args.calib_likelihood),
            vote_likelihood=float(args.vote_likelihood),
            patch_radius=int(args.patch_radius),
            color_dist=float(args.color_dist),
            min_fur_frac=float(args.min_fur_frac),
            min_pixels=int(args.min_pixels),
            dominance=float(args.dominance),
            drop_on_conflict=bool(args.drop_on_conflict),
            manual_mouse0_bgr=manual0,
            manual_mouse1_bgr=manual1,
        )

        if args.dlcOut:
            others = df[~df["camera"].astype(str).isin([args.topCam, args.frontCam, args.sideCam])].copy()
            out_df = pd.concat([others, df_top, df_front, df_side], ignore_index=True)
            out_df.to_csv(args.dlcOut, index=False)
            print(f"[autoCorrect] Wrote corrected DLC CSV: {args.dlcOut}")

    # For overlay, keep only required columns
    cols = ["frame", "individual", "bodypart", "x", "y", "likelihood"]
    df_top = df_top[cols].copy()
    df_front = df_front[cols].copy()
    df_side = df_side[cols].copy()

    print("Annotating top video...")
    if args.sam:
        if args.sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required when using --sam.")
        overlay_video_with_sam(
            args.topIn, args.topOut, df_top,
            edges=edges,
            likelihood_min_draw=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            sam_device=args.sam_device,
            sam_alpha=args.sam_alpha,
            sam_min_lik=args.sam_min_likelihood,
            sam_use_dlc_ids=bool(args.sam_use_dlc_ids),
            sam_every=int(args.sam_every),
            sam_only=bool(args.sam_only),
            mouse0_bgr=manual0,
            mouse1_bgr=manual1,
            sam_calib_frames=int(args.sam_calib_frames),
            sam_calib_stride=int(args.sam_calib_stride),
            sam_max_factor=float(args.sam_max_factor),
            sam_color_sigma_mult=float(args.sam_color_sigma_mult),
            sam_nearby_radius=int(args.sam_nearby_radius),
            sam_nearby_mode=str(args.sam_nearby_mode),
            sam_debug_candidates=bool(args.sam_debug_candidates),
            sam_stats_out=sam_stats_path_for('top'),
            view_name='top',
            sam_selection_mode=str(args.sam_selection_mode),
            number_of_frames=int(args.number_of_frames),
            frame_interval=frame_interval,
        )
    else:
        overlay_video(
            args.topIn, args.topOut, df_top,
            edges=edges,
            likelihood_min=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
        )
    print(f"Wrote: {args.topOut}")

    print("Annotating front video...")
    if args.sam:
        if args.sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required when using --sam.")
        overlay_video_with_sam(
            args.frontIn, args.frontOut, df_front,
            edges=edges,
            likelihood_min_draw=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            sam_device=args.sam_device,
            sam_alpha=args.sam_alpha,
            sam_min_lik=args.sam_min_likelihood,
            sam_use_dlc_ids=bool(args.sam_use_dlc_ids),
            sam_every=int(args.sam_every),
            sam_only=bool(args.sam_only),
            mouse0_bgr=manual0,
            mouse1_bgr=manual1,
            sam_calib_frames=int(args.sam_calib_frames),
            sam_calib_stride=int(args.sam_calib_stride),
            sam_max_factor=float(args.sam_max_factor),
            sam_color_sigma_mult=float(args.sam_color_sigma_mult),
            sam_nearby_radius=int(args.sam_nearby_radius),
            sam_nearby_mode=str(args.sam_nearby_mode),
            sam_debug_candidates=bool(args.sam_debug_candidates),
            sam_stats_out=sam_stats_path_for('front'),
            view_name='front',
            sam_selection_mode=str(args.sam_selection_mode),
            number_of_frames=int(args.number_of_frames),
            frame_interval=frame_interval,
        )
    else:
        overlay_video(
            args.frontIn, args.frontOut, df_front,
            edges=edges,
            likelihood_min=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
        )
    print(f"Wrote: {args.frontOut}")

    print("Annotating side video...")
    if args.sam:
        if args.sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required when using --sam.")
        overlay_video_with_sam(
            args.sideIn, args.sideOut, df_side,
            edges=edges,
            likelihood_min_draw=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            sam_device=args.sam_device,
            sam_alpha=args.sam_alpha,
            sam_min_lik=args.sam_min_likelihood,
            sam_use_dlc_ids=bool(args.sam_use_dlc_ids),
            sam_every=int(args.sam_every),
            sam_only=bool(args.sam_only),
            mouse0_bgr=manual0,
            mouse1_bgr=manual1,
            sam_calib_frames=int(args.sam_calib_frames),
            sam_calib_stride=int(args.sam_calib_stride),
            sam_max_factor=float(args.sam_max_factor),
            sam_color_sigma_mult=float(args.sam_color_sigma_mult),
            sam_nearby_radius=int(args.sam_nearby_radius),
            sam_nearby_mode=str(args.sam_nearby_mode),
            sam_debug_candidates=bool(args.sam_debug_candidates),
            sam_stats_out=sam_stats_path_for('side'),
            view_name='side',
            sam_selection_mode=str(args.sam_selection_mode),
            number_of_frames=int(args.number_of_frames),
            frame_interval=frame_interval,
        )
    else:
        overlay_video(
            args.sideIn, args.sideOut, df_side,
            edges=edges,
            likelihood_min=args.likelihood_min,
            draw_labels=args.draw_labels,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
            force_fps=args.force_fps,
        )
    print(f"Wrote: {args.sideOut}")

    if args.joinedVideo:
        print("Writing joined mosaic video...")
        join_quadrants_from_videos(args.topOut, args.frontOut, args.sideOut, args.joinedVideo, args.force_fps)
        print(f"Wrote: {args.joinedVideo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
