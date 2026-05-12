#!/usr/bin/env python3
"""baPointsOnlyCeres.py

Points-only bundle adjustment using the Ceres backend (mouse_ba.solve_points_only).

No path priors. No rigid template. Each (mouse, frame, joint) point is optimized
independently, but solved in one Ceres problem for convenience and a single solver report.

Default gating prefers reliable 3-view points:
- min_views=3
- max_init_reproj_px=10
- min_ray_angle_deg=2

Output format: coords_3d.csv-style with behavior=0.
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

# optional video backend
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


def _auto_add_build_to_syspath() -> None:
    script_dir = Path(__file__).resolve().parent
    build_dir = script_dir.parent / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))


_auto_add_build_to_syspath()

import mouse_ba  # noqa: E402
from frontend import JOINTS, load_cameras_json, load_dlc_long_csv, default_weight_pack  # noqa: E402


# -----------------------------
# Camera + projection (for diagnostics + videos)
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
# Formatting, metrics, diagnostics
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
                recs.append((int(frames[t]), float(times[t]), int(m), 0, JOINTS[j], float(x), float(y), float(z)))
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
        return pd.DataFrame([{"camera": "ALL", "N": 0, "RMSE_px": np.nan, "MAE_px": np.nan, "Median_px": np.nan, "P95_px": np.nan}])

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
    if _VIDEO_BACKEND is None:
        raise RuntimeError("No video backend available. Install opencv or imageio.")
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
            try:
                w.release()  # type: ignore
            except Exception:
                pass
            try:
                w.close()  # type: ignore
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baPointsOnlyCeres.py")
    p.add_argument("--cameras", required=True)
    p.add_argument("--dlc", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--out", required=True)
    p.add_argument("--gt", default=None)

    p.add_argument("--likelihood-min", type=float, default=0.5, help="Drop DLC observations below this likelihood before BA.")
    p.add_argument("--min-views", type=int, default=3, help="Require at least this many cameras per point (default 3).")
    p.add_argument("--max-init-reproj-px", type=float, default=10.0, help="Gate points by max initial reprojection error (px).")
    p.add_argument("--min-ray-angle-deg", type=float, default=2.0, help="Gate points by minimum pairwise ray angle (deg).")

    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--robust-loss", choices=["huber", "cauchy", "none"], default="huber")
    p.add_argument("--robust-param-px", type=float, default=3.0)

    p.add_argument("--identity-track", action="store_true")
    p.add_argument("--reproj-min-likelihood", type=float, default=0.05)

    p.add_argument("--write-video", action="store_true")
    p.add_argument("--template", default=None, help="Template file (only used to get skeleton edges for video).")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--video-labels", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    cam_quat, cam_trans, cam_intr = load_cameras_json(args.cameras)
    (obs_cam, obs_mouse, obs_joint, obs_frame,
     obs_x, obs_y, obs_l, num_frames) = load_dlc_long_csv(args.dlc)

    M = int(obs_mouse.max()) + 1
    J = len(JOINTS)
    T = int(num_frames)

    wp = default_weight_pack().to_dict()
    wp["likelihood_min"] = float(args.likelihood_min)
    wp["robust_loss"] = str(args.robust_loss)
    wp["robust_param_px"] = float(args.robust_param_px)

    solver_opts = {
        "max_num_iterations": int(args.max_iter),
        "num_threads": int(args.threads),
        "verbose": bool(args.verbose),
        "min_views": int(args.min_views),
        "max_init_reproj_px": float(args.max_init_reproj_px),
        "min_ray_angle_deg": float(args.min_ray_angle_deg),
    }

    out = mouse_ba.solve_points_only(
        cam_quat, cam_trans, cam_intr,
        obs_cam, obs_mouse, obs_joint, obs_frame,
        obs_x, obs_y, obs_l,
        int(T), int(M), int(J),
        wp, solver_opts,
    )
    print(out["summary"])
    print(f"kept_points={out['kept_points']} kept_observations={out['kept_observations']} "
          f"(min_views={out['min_views']}, max_init_reproj_px={out['max_init_reproj_px']}, "
          f"min_ray_angle_deg={out['min_ray_angle_deg']})")

    world = np.asarray(out["points_3d"])
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

    cams, _ = load_scene_cameras(args.cameras)
    rep_est = reprojection_stats(cams, est_df, obs_cam, obs_mouse, obs_joint, obs_frame,
                                 obs_x, obs_y, obs_l, min_likelihood=float(args.reproj_min_likelihood))
    print("\n=== Reprojection error vs DLC (px) for points-only Ceres EST ===")
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
