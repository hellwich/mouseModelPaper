#!/usr/bin/env python3
"""baPointsOnly.py

Ablation baseline: 3D reconstruction using ONLY image observations.
No path priors, no rigid model, no slack priors.

Method:
- For each (mouse_id, frame, joint), collect 2D observations from available cameras
  with likelihood >= --likelihood-min.
- If at least 2 cameras are available: multi-view DLT triangulation (linear).
- Optionally: simple per-frame identity tracking (2 mice) based on head+tail_root continuity.
- Write coords_3d.csv-format output with behavior=0.
- Optional GT comparison and reprojection diagnostics.
- Optional per-camera mp4 rendering (same style as earlier scripts).

This is the cleanest way to answer: "what happens if we remove path+model constraints?"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional video backend
_VIDEO_BACKEND = None
try:
    import cv2  # type: ignore
    _VIDEO_BACKEND = "opencv"
except Exception:
    cv2 = None  # type: ignore

if _VIDEO_BACKEND is None:
    try:
        import imageio.v2 as imageio  # type: ignore
        _VIDEO_BACKEND = "imageio"
    except Exception:
        imageio = None  # type: ignore


# Use the same joint ordering as the rest of the project
try:
    from frontend import JOINTS, JOINT_TO_IDX, load_dlc_long_csv
except Exception as e:
    raise RuntimeError("Run this from the mouseba/python folder (so frontend.py is importable).") from e


# -----------------------------
# Camera + projection
# -----------------------------
class Camera:
    def __init__(self, name: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 R: np.ndarray, t: np.ndarray):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3,)
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.P = self.K @ np.hstack([self.R, self.t.reshape(3, 1)])  # (3,4)

    def project(self, Pw: np.ndarray) -> Tuple[float, float, bool]:
        Pc = self.R @ Pw + self.t
        z = Pc[2]
        if z <= 1e-6:
            return 0.0, 0.0, False
        u = self.fx * (Pc[0] / z) + self.cx
        v = self.fy * (Pc[1] / z) + self.cy
        vis = (0.0 <= u < self.width) and (0.0 <= v < self.height)
        return float(u), float(v), bool(vis)


def load_scene_cameras(cameras_json_path: str | Path) -> Tuple[List[Camera], Tuple[float, float, float]]:
    scene = json.loads(Path(cameras_json_path).read_text(encoding="utf-8"))
    cage = scene["cage"]
    cage_w, cage_d, cage_h = float(cage["width"]), float(cage["depth"]), float(cage["height"])
    cams = []
    for c in scene["cameras"]:
        cams.append(Camera(
            name=c["name"],
            width=c["width"], height=c["height"],
            fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
            R=np.array(c["R"], dtype=np.float64),
            t=np.array(c["t"], dtype=np.float64),
        ))
    return cams, (cage_w, cage_d, cage_h)


# -----------------------------
# Triangulation
# -----------------------------
def triangulate_point(P_list: List[np.ndarray], xy_list: List[Tuple[float, float]]) -> np.ndarray:
    # Linear DLT triangulation for multiple views
    A = []
    for P, (x, y) in zip(P_list, xy_list):
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.stack(A, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


# -----------------------------
# Identity tracking (2 mice)
# -----------------------------
def identity_track_two_mice(world: np.ndarray, anchor_nodes: Tuple[str, str] = ("head", "tail_root")) -> np.ndarray:
    M, T, J, _ = world.shape
    if M != 2:
        return world
    anchor_idx = [JOINTS.index(n) for n in anchor_nodes if n in JOINTS]
    if not anchor_idx:
        return world

    anchor = np.full((M, T, 3), np.nan, dtype=np.float64)
    for m in range(M):
        for t in range(T):
            pts = []
            for j in anchor_idx:
                pts.append(world[m, t, j])
            pts = np.asarray(pts)
            if np.any(np.isnan(pts)):
                continue
            anchor[m, t] = pts.mean(axis=0)

    def dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return 1e6
        d = a - b
        return float(np.sqrt(np.dot(d, d)))

    dp = np.full((T, 2), np.inf)
    prev = np.full((T, 2), -1, dtype=int)
    dp[0, 0] = 0.0
    dp[0, 1] = 0.0

    for t in range(1, T):
        for state in (0, 1):
            map_t = (0, 1) if state == 0 else (1, 0)
            for pst in (0, 1):
                map_prev = (0, 1) if pst == 0 else (1, 0)
                c = dist(anchor[map_t[0], t], anchor[map_prev[0], t-1]) + dist(anchor[map_t[1], t], anchor[map_prev[1], t-1])
                v = dp[t-1, pst] + c
                if v < dp[t, state]:
                    dp[t, state] = v
                    prev[t, state] = pst

    state = int(np.argmin(dp[T-1]))
    states = [state]
    for t in range(T-1, 0, -1):
        state = prev[t, state]
        states.append(state)
    states = list(reversed(states))

    out = world.copy()
    swapped = 0
    for t, st in enumerate(states):
        if st == 1:
            out[:, t] = out[::-1, t]
            swapped += 1
    print(f"[identity-track] swapped {swapped}/{T} frames ({100.0*swapped/T:.1f}%).")
    return out


# -----------------------------
# Output formatting
# -----------------------------
def coords_df_from_world(world: np.ndarray, fps: float) -> pd.DataFrame:
    M, T, J, _ = world.shape
    frames = np.arange(T, dtype=np.int32)
    times = frames.astype(np.float64) / float(fps)
    recs = []
    for m in range(M):
        for t in range(T):
            for j in range(J):
                x, y, z = world[m, t, j]
                recs.append((
                    int(frames[t]),
                    float(times[t]),
                    int(m),
                    0,  # behavior placeholder
                    JOINTS[j],
                    float(x) if np.isfinite(x) else np.nan,
                    float(y) if np.isfinite(y) else np.nan,
                    float(z) if np.isfinite(z) else np.nan,
                ))
    return pd.DataFrame(recs, columns=["frame", "time", "mouse_id", "behavior", "node", "x", "y", "z"])


def compare_to_ground_truth(est_df: pd.DataFrame, gt_path: str | Path) -> pd.DataFrame:
    gt_df = pd.read_csv(gt_path)
    keys = ["frame", "mouse_id", "node"]
    m = est_df.merge(gt_df[keys + ["x", "y", "z"]], on=keys, suffixes=("_est", "_gt"), how="inner")
    m = m.dropna(subset=["x_est", "y_est", "z_est", "x_gt", "y_gt", "z_gt"])
    dx = m["x_est"].to_numpy() - m["x_gt"].to_numpy()
    dy = m["y_est"].to_numpy() - m["y_gt"].to_numpy()
    dz = m["z_est"].to_numpy() - m["z_gt"].to_numpy()
    err = np.sqrt(dx*dx + dy*dy + dz*dz)
    return pd.DataFrame([{
        "N": int(len(err)),
        "RMSE_mm": float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "MAE_mm": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "Median_mm": float(np.median(err)) if len(err) else np.nan,
        "P95_mm": float(np.percentile(err, 95)) if len(err) else np.nan,
    }])


def reprojection_stats(
    cams: List[Camera],
    coords_df: pd.DataFrame,
    obs_cam: np.ndarray, obs_mouse: np.ndarray, obs_joint: np.ndarray, obs_frame: np.ndarray,
    obs_x: np.ndarray, obs_y: np.ndarray, obs_l: np.ndarray,
    min_likelihood: float,
) -> pd.DataFrame:
    key_to_xyz = {}
    for row in coords_df.itertuples(index=False):
        key_to_xyz[(int(row.frame), int(row.mouse_id), str(row.node))] = (float(row.x), float(row.y), float(row.z))

    cam_index_to_cam = {i: cams[i] for i in range(len(cams))}
    errs = []
    for i in range(len(obs_x)):
        if float(obs_l[i]) < min_likelihood:
            continue
        fr = int(obs_frame[i])
        mid = int(obs_mouse[i])
        jidx = int(obs_joint[i])
        node = JOINTS[jidx]
        key = (fr, mid, node)
        if key not in key_to_xyz:
            continue
        X = key_to_xyz[key]
        if not (np.isfinite(X[0]) and np.isfinite(X[1]) and np.isfinite(X[2])):
            continue
        Pw = np.array(X, dtype=np.float64)
        cam = cam_index_to_cam[int(obs_cam[i])]
        u, v, vis = cam.project(Pw)
        if not vis:
            continue
        dx = u - float(obs_x[i])
        dy = v - float(obs_y[i])
        errs.append((cam.name, math.sqrt(dx*dx + dy*dy)))

    if not errs:
        return pd.DataFrame([{ "camera": "ALL", "N": 0, "RMSE_px": np.nan, "MAE_px": np.nan, "Median_px": np.nan, "P95_px": np.nan }])

    df = pd.DataFrame(errs, columns=["camera", "err_px"])
    overall = pd.DataFrame([{
        "camera": "ALL",
        "N": int(len(df)),
        "RMSE_px": float(np.sqrt(np.mean(df["err_px"]**2))),
        "MAE_px": float(df["err_px"].mean()),
        "Median_px": float(df["err_px"].median()),
        "P95_px": float(np.percentile(df["err_px"], 95)),
    }])
    per_cam = df.groupby("camera")["err_px"].agg(
        N="count",
        RMSE_px=lambda x: float(np.sqrt(np.mean(np.square(x)))),
        MAE_px="mean",
        Median_px="median",
        P95_px=lambda x: float(np.percentile(x, 95)),
    ).reset_index()
    return pd.concat([overall, per_cam], ignore_index=True)


# -----------------------------
# Video rendering
# -----------------------------
def parse_edges_from_template(path: str | Path) -> List[Tuple[str, str]]:
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


def render_frame(cam: Camera, edges: List[Tuple[str, str]], world_nodes_by_mouse: List[Dict[str, np.ndarray]],
                 cage_w: float, cage_d: float, cage_h: float, draw_labels: bool = False) -> np.ndarray:
    img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    if cv2 is None:
        return img

    corners = [(x, y, z) for x in (0.0, cage_w) for y in (0.0, cage_d) for z in (0.0, cage_h)]
    cage_pts = [np.array([x, y, z], dtype=np.float64) for x, y, z in corners]
    corner_index = {corners[i]: i for i in range(len(corners))}
    def idx(x, y, z): return corner_index[(x, y, z)]
    box_edges = [
        (idx(0.0, 0.0, 0.0), idx(cage_w, 0.0, 0.0)),
        (idx(0.0, 0.0, 0.0), idx(0.0, cage_d, 0.0)),
        (idx(cage_w, 0.0, 0.0), idx(cage_w, cage_d, 0.0)),
        (idx(0.0, cage_d, 0.0), idx(cage_w, cage_d, 0.0)),
        (idx(0.0, 0.0, cage_h), idx(cage_w, 0.0, cage_h)),
        (idx(0.0, 0.0, cage_h), idx(0.0, cage_d, cage_h)),
        (idx(cage_w, 0.0, cage_h), idx(cage_w, cage_d, cage_h)),
        (idx(0.0, cage_d, cage_h), idx(cage_w, cage_d, cage_h)),
        (idx(0.0, 0.0, 0.0), idx(0.0, 0.0, cage_h)),
        (idx(cage_w, 0.0, 0.0), idx(cage_w, 0.0, cage_h)),
        (idx(0.0, cage_d, 0.0), idx(0.0, cage_d, cage_h)),
        (idx(cage_w, cage_d, 0.0), idx(cage_w, cage_d, cage_h)),
    ]
    cage_uv = []
    for p in cage_pts:
        u, v, vis = cam.project(p)
        cage_uv.append((u, v, vis))
    for a, b in box_edges:
        ua, va, visa = cage_uv[a]
        ub, vb, visb = cage_uv[b]
        if visa and visb:
            cv2.line(img, (int(ua), int(va)), (int(ub), int(vb)), (60, 60, 60), 1, cv2.LINE_AA)

    colors = [(255, 255, 255), (80, 220, 80)]
    node_cols = [(255, 255, 255), (120, 255, 120)]

    for mi, world_nodes in enumerate(world_nodes_by_mouse):
        col = colors[mi % len(colors)]
        coln = node_cols[mi % len(node_cols)]
        for a, b in edges:
            if a not in world_nodes or b not in world_nodes:
                continue
            ua, va, visa = cam.project(world_nodes[a])
            ub, vb, visb = cam.project(world_nodes[b])
            if visa and visb:
                cv2.line(img, (int(ua), int(va)), (int(ub), int(vb)), col, 2, cv2.LINE_AA)
        for label, Pw in world_nodes.items():
            u, v, vis = cam.project(Pw)
            if not vis:
                continue
            cv2.circle(img, (int(u), int(v)), 4, coln, -1, cv2.LINE_AA)
            if draw_labels:
                cv2.putText(img, f"{mi}:{label}", (int(u) + 5, int(v) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, coln, 1, cv2.LINE_AA)
    return img


def write_videos(cameras_json_path: str | Path, template_path: str | Path, coords_df: pd.DataFrame,
                 out_dir: str | Path, fps: float, draw_labels: bool = False) -> None:
    cams, (cage_w, cage_d, cage_h) = load_scene_cameras(cameras_json_path)
    edges = parse_edges_from_template(template_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(coords_df["frame"].unique())
    mice = sorted(coords_df["mouse_id"].unique())

    grouped: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for (fr, mid), g in coords_df.groupby(["frame", "mouse_id"]):
        d = {}
        for _, row in g.iterrows():
            if not (np.isfinite(row["x"]) and np.isfinite(row["y"]) and np.isfinite(row["z"])):
                continue
            d[row["node"]] = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        grouped[(int(fr), int(mid))] = d

    writers = {}
    if _VIDEO_BACKEND == "opencv":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        for cam in cams:
            path = str(out_dir / f"{cam.name}.mp4")
            writers[cam.name] = cv2.VideoWriter(path, fourcc, float(fps), (cam.width, cam.height))  # type: ignore
    else:
        for cam in cams:
            path = str(out_dir / f"{cam.name}.mp4")
            writers[cam.name] = imageio.get_writer(path, fps=float(fps))  # type: ignore

    try:
        for fr in frames:
            world_nodes_by_mouse = [grouped.get((int(fr), int(mid)), {}) for mid in mice]
            for cam in cams:
                img = render_frame(cam, edges, world_nodes_by_mouse, cage_w, cage_d, cage_h, draw_labels=draw_labels)
                if _VIDEO_BACKEND == "opencv":
                    writers[cam.name].write(img)  # type: ignore
                else:
                    writers[cam.name].append_data(img[..., ::-1])  # BGR->RGB
    finally:
        for w in writers.values():
            try: w.release()  # type: ignore
            except Exception: pass
            try: w.close()  # type: ignore
            except Exception: pass


# -----------------------------
# Main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baPointsOnly.py")
    p.add_argument("--cameras", required=True, help="Path to cameras.json")
    p.add_argument("--dlc", required=True, help="Path to dlc_long.csv")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--out", required=True, help="Output coords_3d.csv path")
    p.add_argument("--gt", default=None, help="Optional GT coords_3d.csv")
    p.add_argument("--likelihood-min", type=float, default=0.5,
                   help="Only use DLC observations with likelihood >= this (default 0.5)")
    p.add_argument("--identity-track", action="store_true")
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--template", default=None, help="Template file for skeleton edges when writing video")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--video-labels", action="store_true")
    p.add_argument("--reproj-min-likelihood", type=float, default=0.05,
                   help="Min likelihood for reprojection diagnostics (default 0.05)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    cams, _ = load_scene_cameras(args.cameras)

    (obs_cam, obs_mouse, obs_joint, obs_frame,
     obs_x, obs_y, obs_l, num_frames) = load_dlc_long_csv(args.dlc)

    M = int(obs_mouse.max()) + 1
    T = int(num_frames)
    J = len(JOINTS)

    # Build obs index: key -> list of (cam, x, y)
    key_to_obs: Dict[Tuple[int, int, int], List[Tuple[int, float, float]]] = {}
    for i in range(len(obs_x)):
        if float(obs_l[i]) < float(args.likelihood_min):
            continue
        key = (int(obs_mouse[i]), int(obs_frame[i]), int(obs_joint[i]))
        key_to_obs.setdefault(key, []).append((int(obs_cam[i]), float(obs_x[i]), float(obs_y[i])))

    world = np.full((M, T, J, 3), np.nan, dtype=np.float64)
    used = 0
    for m in range(M):
        for t in range(T):
            for j in range(J):
                obs = key_to_obs.get((m, t, j), [])
                if len(obs) < 2:
                    continue
                # Use at most one observation per camera
                by_cam = {}
                for c, x, y in obs:
                    if c not in by_cam:
                        by_cam[c] = (x, y)
                if len(by_cam) < 2:
                    continue
                P_list = []
                xy_list = []
                for c, (x, y) in sorted(by_cam.items()):
                    P_list.append(cams[c].P)
                    xy_list.append((x, y))
                X = triangulate_point(P_list, xy_list)
                world[m, t, j] = X
                used += 1

    print(f"Triangulated {used}/{M*T*J} joint instances (>=2 cameras, likelihood>={args.likelihood_min}).")

    if args.identity_track:
        world = identity_track_two_mice(world)

    est_df = coords_df_from_world(world, fps=float(args.fps))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    est_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    if args.gt:
        summary = compare_to_ground_truth(est_df, args.gt)
        print("\n=== 3D Accuracy vs GT (mm) ===")
        print(summary.to_string(index=False))

    rep_est = reprojection_stats(cams, est_df, obs_cam, obs_mouse, obs_joint, obs_frame,
                                 obs_x, obs_y, obs_l, min_likelihood=float(args.reproj_min_likelihood))
    print("\n=== Reprojection error vs DLC (px) for triangulation EST ===")
    print(rep_est.to_string(index=False))

    if args.gt:
        gt_df = pd.read_csv(args.gt)
        rep_gt = reprojection_stats(cams, gt_df, obs_cam, obs_mouse, obs_joint, obs_frame,
                                    obs_x, obs_y, obs_l, min_likelihood=float(args.reproj_min_likelihood))
        print("\n=== Reprojection error vs DLC (px) for GT ===")
        print(rep_gt.to_string(index=False))

    if args.write_video:
        if args.template is None:
            raise RuntimeError("--write-video requires --template to draw skeleton edges.")
        video_dir = Path(args.video_dir) if args.video_dir else out_path.parent
        write_videos(args.cameras, args.template, est_df, video_dir, fps=float(args.fps), draw_labels=bool(args.video_labels))
        print(f"Wrote videos to: {video_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
