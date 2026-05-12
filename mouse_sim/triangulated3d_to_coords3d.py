#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Helpers / normalization
# ----------------------------

def nlabel(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt((v * v).sum()) + eps)


def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        raise ValueError("smoothing window must be odd")
    r = k // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        a = max(0, i - r)
        b = min(len(x), i + r + 1)
        out[i] = float(np.mean(x[a:b]))
    return out


def interpolate_nan_1d(y: np.ndarray) -> np.ndarray:
    out = y.astype(np.float64).copy()
    n = len(out)
    idx = np.arange(n)
    good = np.isfinite(out)
    if good.all():
        return out
    if not good.any():
        return out
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Observation:
    point: np.ndarray   # (3,)
    kind: str           # e.g. pair/triplet
    num_obs: int


@dataclass
class FitState:
    rotation: np.ndarray          # (3,3)
    translation: np.ndarray       # (3,)
    used_nodes: List[str]
    rank: int
    mode: str

    def apply(self, p_model: np.ndarray) -> np.ndarray:
        return (self.rotation @ p_model.astype(np.float64)) + self.translation

    def inverse_apply(self, p_world: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (p_world.astype(np.float64) - self.translation)


# ----------------------------
# Parsers
# ----------------------------

def parse_mouse_graph(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model: Dict[str, np.ndarray] = {}
    section = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.upper()
                continue
            if section == "[NODES]":
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Malformed node line in {path!r}: {line!r}")
                name = nlabel(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                model[name] = xyz
    if not model:
        raise ValueError(f"No [NODES] parsed from {path}")
    return model


def load_triangulated_3d(path: str) -> Tuple[List[int], List[int], Dict[Tuple[int, int], Dict[str, Observation]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    by_fm: Dict[Tuple[int, int], Dict[str, Observation]] = defaultdict(dict)
    frames: set[int] = set()
    mouse_ids: set[int] = set()

    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"frame", "bodypart", "mouse_id", "X", "Y", "Z", "kind", "num_obs"}
        have = set(r.fieldnames or [])
        if not required.issubset(have):
            raise ValueError(f"{path} missing required columns. Have: {r.fieldnames}")
        for rr in r:
            frame = int(rr["frame"])
            mouse_id = int(rr["mouse_id"])
            node = nlabel(rr["bodypart"])
            point = np.array([float(rr["X"]), float(rr["Y"]), float(rr["Z"])], dtype=np.float64)
            if not np.all(np.isfinite(point)):
                continue
            frames.add(frame)
            mouse_ids.add(mouse_id)
            by_fm[(frame, mouse_id)][node] = Observation(
                point=point,
                kind=nlabel(rr.get("kind", "")),
                num_obs=int(rr.get("num_obs", 0) or 0),
            )

    if not frames:
        raise ValueError(f"{path}: no valid rows")
    return sorted(frames), sorted(mouse_ids), by_fm


def load_ground_truth_behavior(path: str) -> Dict[Tuple[int, int], int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: Dict[Tuple[int, int], int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"frame", "mouse_id", "behavior"}
        have = set(r.fieldnames or [])
        if not required.issubset(have):
            raise ValueError(f"{path} missing required ground-truth columns. Have: {r.fieldnames}")
        for rr in r:
            out[(int(rr["frame"]), int(rr["mouse_id"]))] = int(rr["behavior"])
    return out


# ----------------------------
# Rigid alignment
# ----------------------------

def obs_weight(obs: Observation, pair_weight: float, triplet_weight: float) -> float:
    if obs.kind == "triplet":
        base = triplet_weight
    elif obs.kind == "pair":
        base = pair_weight
    else:
        base = 1.0
    # keep weighting simple; num_obs can only help weakly
    return float(base * max(obs.num_obs, 1))


def weighted_rigid_transform(
    model_pts: np.ndarray,
    world_pts: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Weighted Kabsch.
    model_pts/world_pts: (N,3), weights: (N,)
    Returns rotation R, translation t, singular values, covariance rank.
    Such that world ~= R @ model + t.
    """
    w = weights.astype(np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("weights sum to zero")
    w = w / wsum

    mu_model = np.sum(model_pts * w[:, None], axis=0)
    mu_world = np.sum(world_pts * w[:, None], axis=0)

    xm = model_pts - mu_model
    xw = world_pts - mu_world
    H = xm.T @ (w[:, None] * xw)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_world - (R @ mu_model)
    rank = int(np.sum(S > max(S[0] if len(S) else 0.0, 1.0) * 1e-8)) if len(S) else 0
    return R, t, S, rank


def fit_from_nodes(
    observations: Dict[str, Observation],
    model_nodes: Dict[str, np.ndarray],
    node_names: Sequence[str],
    pair_weight: float,
    triplet_weight: float,
) -> Optional[Tuple[FitState, np.ndarray]]:
    used: List[str] = []
    mpts: List[np.ndarray] = []
    wpts: List[np.ndarray] = []
    ws: List[float] = []
    for name in node_names:
        obs = observations.get(name)
        if obs is None or name not in model_nodes:
            continue
        used.append(name)
        mpts.append(model_nodes[name])
        wpts.append(obs.point)
        ws.append(obs_weight(obs, pair_weight, triplet_weight))
    if len(used) < 3:
        return None
    model_arr = np.stack(mpts, axis=0)
    world_arr = np.stack(wpts, axis=0)
    weights = np.asarray(ws, dtype=np.float64)
    R, t, S, rank = weighted_rigid_transform(model_arr, world_arr, weights)
    state = FitState(rotation=R, translation=t, used_nodes=used, rank=rank, mode="fit")
    return state, S


def is_degenerate(state: FitState, singvals: np.ndarray) -> bool:
    if len(state.used_nodes) < 3:
        return True
    if len(singvals) < 2:
        return True
    s0 = float(singvals[0]) if singvals.size else 0.0
    s1 = float(singvals[1]) if singvals.size > 1 else 0.0
    if s0 <= 1e-12:
        return True
    # if the second singular value is tiny, geometry is close to collinear
    return (state.rank < 2) or (s1 / s0 < 1e-3)


def build_fallback_state(
    observations: Dict[str, Observation],
    model_nodes: Dict[str, np.ndarray],
    prev_state: Optional[FitState],
) -> FitState:
    if prev_state is not None:
        used = [n for n in observations.keys() if n in model_nodes]
        if used:
            mu_model = np.mean([model_nodes[n] for n in used], axis=0)
            mu_world = np.mean([observations[n].point for n in used], axis=0)
            t = mu_world - (prev_state.rotation @ mu_model)
            return FitState(
                rotation=prev_state.rotation.copy(),
                translation=t,
                used_nodes=list(used),
                rank=0,
                mode="prev_rotation",
            )
        return FitState(
            rotation=prev_state.rotation.copy(),
            translation=prev_state.translation.copy(),
            used_nodes=[],
            rank=0,
            mode="prev_state",
        )

    # First frame, too little information: use identity and align tail_root if possible, else centroid.
    R = np.eye(3, dtype=np.float64)
    if "tail_root" in observations and "tail_root" in model_nodes:
        t = observations["tail_root"].point - model_nodes["tail_root"]
        used_nodes = ["tail_root"]
    elif observations:
        used_nodes = [n for n in observations.keys() if n in model_nodes]
        if used_nodes:
            mu_model = np.mean([model_nodes[n] for n in used_nodes], axis=0)
            mu_world = np.mean([observations[n].point for n in used_nodes], axis=0)
            t = mu_world - mu_model
        else:
            used_nodes = []
            t = np.zeros(3, dtype=np.float64)
    else:
        used_nodes = []
        t = np.zeros(3, dtype=np.float64)
    return FitState(rotation=R, translation=t, used_nodes=used_nodes, rank=0, mode="identity")


def estimate_fit_state(
    observations: Dict[str, Observation],
    model_nodes: Dict[str, np.ndarray],
    prev_state: Optional[FitState],
    pair_weight: float,
    triplet_weight: float,
    min_fit_points: int,
) -> FitState:
    rigid_pref = [
        "head",
        "nose_tip",
        "left_ear_tip",
        "right_ear_tip",
        "tail_root",
        "tail_tip",
    ]
    paws = [
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
    ]
    all_known = [n for n in rigid_pref + paws if n in model_nodes]

    available_pref = [n for n in rigid_pref if n in observations and n in model_nodes]
    available_all = [n for n in all_known if n in observations]

    # Try preferred rigid landmarks first if enough points.
    if len(available_pref) >= min_fit_points:
        res = fit_from_nodes(observations, model_nodes, available_pref, pair_weight, triplet_weight)
        if res is not None:
            st, S = res
            if not is_degenerate(st, S):
                st.mode = "preferred"
                return st
            # supplement with paws if preferred set is geometrically weak
            if len(available_all) >= min_fit_points and len(available_all) > len(available_pref):
                res2 = fit_from_nodes(observations, model_nodes, available_all, pair_weight, triplet_weight)
                if res2 is not None:
                    st2, _S2 = res2
                    st2.mode = "preferred_plus_paws"
                    return st2
            if prev_state is not None:
                fb = build_fallback_state({n: observations[n] for n in available_pref}, model_nodes, prev_state)
                fb.mode = "degenerate_prev_rotation"
                return fb
            st.mode = "degenerate_preferred"
            return st

    # If too few preferred nodes, fall back to any observed nodes (including paws).
    if len(available_all) >= min_fit_points:
        res = fit_from_nodes(observations, model_nodes, available_all, pair_weight, triplet_weight)
        if res is not None:
            st, S = res
            if is_degenerate(st, S) and prev_state is not None:
                fb = build_fallback_state({n: observations[n] for n in available_all}, model_nodes, prev_state)
                fb.mode = "degenerate_all_prev_rotation"
                return fb
            st.mode = "all_nodes"
            return st

    # Last resort: preserve continuity rather than emit zeros.
    return build_fallback_state(observations, model_nodes, prev_state)


# ----------------------------
# Imputation / smoothing
# ----------------------------

def compute_fit_states(
    frames: Sequence[int],
    mouse_ids: Sequence[int],
    by_fm: Dict[Tuple[int, int], Dict[str, Observation]],
    model_nodes: Dict[str, np.ndarray],
    pair_weight: float,
    triplet_weight: float,
    min_fit_points: int,
) -> Dict[Tuple[int, int], FitState]:
    out: Dict[Tuple[int, int], FitState] = {}
    for mouse_id in mouse_ids:
        prev: Optional[FitState] = None
        for frame in frames:
            obs = by_fm.get((frame, mouse_id), {})
            st = estimate_fit_state(
                observations=obs,
                model_nodes=model_nodes,
                prev_state=prev,
                pair_weight=pair_weight,
                triplet_weight=triplet_weight,
                min_fit_points=min_fit_points,
            )
            out[(frame, mouse_id)] = st
            prev = st
    return out


def build_body_frame_paw_tracks(
    frames: Sequence[int],
    mouse_ids: Sequence[int],
    by_fm: Dict[Tuple[int, int], Dict[str, Observation]],
    fit_states: Dict[Tuple[int, int], FitState],
    paw_nodes: Sequence[str],
    smooth_window: int,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Returns body-frame paw tracks for all frames/mice/nodes, filled by interpolation
    and optional moving-average smoothing.
    """
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    for mouse_id in mouse_ids:
        paw_tracks: Dict[str, np.ndarray] = {
            n: np.full((len(frames), 3), np.nan, dtype=np.float64) for n in paw_nodes
        }
        for ti, frame in enumerate(frames):
            obs = by_fm.get((frame, mouse_id), {})
            st = fit_states[(frame, mouse_id)]
            for paw in paw_nodes:
                if paw in obs:
                    paw_tracks[paw][ti] = st.inverse_apply(obs[paw].point)

        for paw in paw_nodes:
            arr = paw_tracks[paw]
            good = np.isfinite(arr).all(axis=1)
            if not good.any():
                continue
            filled = arr.copy()
            for d in range(3):
                filled[:, d] = interpolate_nan_1d(filled[:, d])
                if smooth_window > 1:
                    filled[:, d] = moving_average_1d(filled[:, d], smooth_window)
            paw_tracks[paw] = filled

        for ti, frame in enumerate(frames):
            out[(frame, mouse_id)] = {paw: paw_tracks[paw][ti].copy() for paw in paw_nodes}
    return out


def reconstruct_dense_rows(
    frames: Sequence[int],
    mouse_ids: Sequence[int],
    node_order: Sequence[str],
    by_fm: Dict[Tuple[int, int], Dict[str, Observation]],
    fit_states: Dict[Tuple[int, int], FitState],
    model_nodes: Dict[str, np.ndarray],
    paw_body_tracks: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    fps: float,
    ground_truth: Optional[Dict[Tuple[int, int], int]],
) -> Tuple[List[List[object]], Dict[str, Dict[str, int]]]:
    rows: List[List[object]] = []
    stats: Dict[str, Dict[str, int]] = {
        "observed": defaultdict(int),
        "imputed": defaultdict(int),
    }
    paw_nodes = {"left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"}

    prev_world: Dict[Tuple[int, int, str], np.ndarray] = {}

    for frame in frames:
        time_s = float(frame) / float(fps)
        for mouse_id in mouse_ids:
            obs = by_fm.get((frame, mouse_id), {})
            st = fit_states[(frame, mouse_id)]
            beh = None if ground_truth is None else ground_truth.get((frame, mouse_id), 0)
            for node in node_order:
                if node in obs:
                    p = obs[node].point.astype(np.float64)
                    stats["observed"][node] += 1
                else:
                    if node in paw_nodes:
                        local = paw_body_tracks.get((frame, mouse_id), {}).get(node)
                        if local is not None and np.all(np.isfinite(local)):
                            p = st.apply(local)
                        elif (frame - 1, mouse_id, node) in prev_world:
                            p = prev_world[(frame - 1, mouse_id, node)].copy()
                        else:
                            p = st.apply(model_nodes[node])
                    else:
                        p = st.apply(model_nodes[node])
                    stats["imputed"][node] += 1
                prev_world[(frame, mouse_id, node)] = p.copy()
                row = [
                    int(frame),
                    f"{time_s:.6f}",
                    int(mouse_id),
                ]
                if ground_truth is not None:
                    row.append(int(beh if beh is not None else 0))
                row.extend([
                    node,
                    f"{float(p[0]):.6f}",
                    f"{float(p[1]):.6f}",
                    f"{float(p[2]):.6f}",
                ])
                rows.append(row)
    return rows, stats


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        "Convert sparse triangulated 3D mouse keypoints into dense coords_3d.csv format compatible with predict_behavior_pair_xlstm.py"
    )
    ap.add_argument("--triangulated-3d", required=True, help="Input triangulated3d.csv")
    ap.add_argument("--mouse-graph", required=True, help="Rigid mouse graph .txt with canonical node coordinates")
    ap.add_argument("--out-csv", required=True, help="Output coords_3d.csv path")
    ap.add_argument("--fps", type=float, default=10.0, help="Frames per second used to synthesize the time column")
    ap.add_argument("--ground-truth", default=None,
                    help="Optional simulated coords_3d.csv from which behavior labels are copied by (frame,mouse_id)")
    ap.add_argument("--pair-weight", type=float, default=1.0,
                    help="Weight for pair-based triangulated points in rigid fitting")
    ap.add_argument("--triplet-weight", type=float, default=3.0,
                    help="Weight for triplet-based triangulated points in rigid fitting")
    ap.add_argument("--min-fit-points", type=int, default=3,
                    help="Minimum number of observed points required to attempt a rigid fit")
    ap.add_argument("--paw-smooth-window", type=int, default=5,
                    help="Odd temporal smoothing window for paw trajectories in body coordinates (1 disables)")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.min_fit_points < 3:
        raise ValueError("--min-fit-points must be >= 3")
    if args.paw_smooth_window < 1 or args.paw_smooth_window % 2 == 0:
        raise ValueError("--paw-smooth-window must be an odd integer >= 1")

    model_nodes = parse_mouse_graph(args.mouse_graph)
    node_order = list(model_nodes.keys())
    frames, mouse_ids, by_fm = load_triangulated_3d(args.triangulated_3d)

    if len(mouse_ids) < 2:
        raise ValueError(f"Expected at least 2 mouse ids, found {mouse_ids}")
    mouse_ids = mouse_ids[:2]

    ground_truth = None
    if args.ground_truth is not None:
        ground_truth = load_ground_truth_behavior(args.ground_truth)

    fit_states = compute_fit_states(
        frames=frames,
        mouse_ids=mouse_ids,
        by_fm=by_fm,
        model_nodes=model_nodes,
        pair_weight=args.pair_weight,
        triplet_weight=args.triplet_weight,
        min_fit_points=args.min_fit_points,
    )

    paw_nodes = [
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
    ]
    paw_tracks = build_body_frame_paw_tracks(
        frames=frames,
        mouse_ids=mouse_ids,
        by_fm=by_fm,
        fit_states=fit_states,
        paw_nodes=paw_nodes,
        smooth_window=args.paw_smooth_window,
    )

    rows, stats = reconstruct_dense_rows(
        frames=frames,
        mouse_ids=mouse_ids,
        node_order=node_order,
        by_fm=by_fm,
        fit_states=fit_states,
        model_nodes=model_nodes,
        paw_body_tracks=paw_tracks,
        fps=args.fps,
        ground_truth=ground_truth,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["frame", "time", "mouse_id"]
        if ground_truth is not None:
            header.append("behavior")
        header.extend(["node", "x", "y", "z"])
        w.writerow(header)
        w.writerows(rows)

    mode_counts: Dict[str, int] = defaultdict(int)
    for st in fit_states.values():
        mode_counts[st.mode] += 1

    print(f"Wrote: {args.out_csv}")
    print(f"Frames: {len(frames)}  Mice: {mouse_ids}  Nodes/frame/mouse: {len(node_order)}")
    print("Fit modes:")
    for mode in sorted(mode_counts):
        print(f"  {mode}: {mode_counts[mode]}")
    print("Per-node observation / imputation counts:")
    for node in node_order:
        obs_n = stats["observed"].get(node, 0)
        imp_n = stats["imputed"].get(node, 0)
        print(f"  {node:16s} observed={obs_n:5d}  imputed={imp_n:5d}")


if __name__ == "__main__":
    main()
