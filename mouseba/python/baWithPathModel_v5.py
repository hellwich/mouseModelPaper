#!/usr/bin/env python3
"""baWithPathModel_v5.py

Adds on top of v4:
- Reprojection diagnostics:
    * reprojection error of estimated 3D back into each camera vs DLC observations
    * (optional) reprojection error of provided GT coords_3d.csv vs DLC observations
  This quickly tells whether:
    - the GT file matches the DLC file/cameras, and
    - the BA solution actually fits the 2D input.
- Identity tracking (to reduce mouse-id switches) without GT:
    * --identity-track performs a simple 2-mouse Viterbi assignment over time
      based on anchor-joint continuity (head + tail_root).
  This affects output CSV and rendered videos (so you can visually judge).

Other features retained:
- quat order auto-test (wxyz vs xyzw) when --gt is provided
- pose mode auto-test (body_to_world vs world_to_body) when --gt is provided
- optional mp4 rendering per camera (cage + skeleton)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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


def _auto_add_build_to_syspath() -> None:
    script_dir = Path(__file__).resolve().parent
    build_dir = script_dir.parent / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))


_auto_add_build_to_syspath()

from frontend import (
    JOINTS,
    load_cameras_json,
    load_template_points,
    load_dlc_long_csv,
    init_poses_from_triangulation,
    default_weight_pack,
)

import mouse_ba


# -----------------------------
# Quaternion helpers
# -----------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return q / n


def quat_to_rotmat(q: np.ndarray, order: str) -> np.ndarray:
    q = quat_normalize(q)
    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    elif order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = ww + xx - yy - zz
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)

    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = ww - xx + yy - zz
    R[..., 1, 2] = 2.0 * (yz - wx)

    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = ww - xx - yy + zz
    return R


# -----------------------------
# World coordinates from solver state
# -----------------------------
def compute_world_points(
    template_pts: np.ndarray,           # (J,3)
    mouse_quat: np.ndarray,             # (M,T,4)
    mouse_trans: np.ndarray,            # (M,T,3)
    slack: np.ndarray,                  # (M,T,J,3)
    quat_order: str,
    pose_mode: str,
) -> np.ndarray:
    """Return world points (M,T,J,3)."""
    R = quat_to_rotmat(mouse_quat, quat_order)   # (M,T,3,3)
    local = template_pts[None, None, :, :] + slack
    if pose_mode == "body_to_world":
        world = (R[..., None, :, :] @ local[..., None]).squeeze(-1) + mouse_trans[..., None, :]
    elif pose_mode == "world_to_body":
        Rt = np.swapaxes(R, -1, -2)
        world = (Rt[..., None, :, :] @ local[..., None]).squeeze(-1) + mouse_trans[..., None, :]
    else:
        raise ValueError(f"Unknown pose_mode: {pose_mode}")
    return world


def build_coords_df_from_world(world: np.ndarray, fps: float, behavior_default: int = 0) -> pd.DataFrame:
    M, T, J, _ = world.shape
    frames = np.arange(T, dtype=np.int32)
    times = frames.astype(np.float64) / float(fps)

    recs = []
    for m in range(M):
        for t in range(T):
            for j in range(J):
                recs.append((
                    int(frames[t]),
                    float(times[t]),
                    int(m),
                    int(behavior_default),
                    JOINTS[j],
                    float(world[m, t, j, 0]),
                    float(world[m, t, j, 1]),
                    float(world[m, t, j, 2]),
                ))
    return pd.DataFrame(recs, columns=["frame", "time", "mouse_id", "behavior", "node", "x", "y", "z"])


# -----------------------------
# GT comparison
# -----------------------------
def compare_to_ground_truth(est_df: pd.DataFrame, gt_path: str | Path) -> Dict[str, pd.DataFrame]:
    gt_df = pd.read_csv(gt_path)
    required = {"frame", "mouse_id", "node", "x", "y", "z"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(f"Ground truth file missing required columns: {sorted(missing)}")

    keys = ["frame", "mouse_id", "node"]
    m = est_df.merge(gt_df[keys + ["x", "y", "z"]], on=keys, suffixes=("_est", "_gt"), how="inner")
    if len(m) == 0:
        raise ValueError("No matching rows between estimate and ground truth (check node names / frame count).")

    dx = m["x_est"].to_numpy() - m["x_gt"].to_numpy()
    dy = m["y_est"].to_numpy() - m["y_gt"].to_numpy()
    dz = m["z_est"].to_numpy() - m["z_gt"].to_numpy()
    err = np.sqrt(dx*dx + dy*dy + dz*dz)
    m["err_mm"] = err

    overall = pd.DataFrame([{
        "N": int(len(err)),
        "RMSE_mm": float(np.sqrt(np.mean(err**2))),
        "MAE_mm": float(np.mean(np.abs(err))),
        "Median_mm": float(np.median(err)),
        "P95_mm": float(np.percentile(err, 95)),
        "P99_mm": float(np.percentile(err, 99)),
    }])

    offsets = pd.DataFrame([{
        "mean_dx": float(dx.mean()),
        "mean_dy": float(dy.mean()),
        "mean_dz": float(dz.mean()),
        "std_dx": float(dx.std()),
        "std_dy": float(dy.std()),
        "std_dz": float(dz.std()),
    }])

    per_node = (
        m.groupby("node")["err_mm"]
         .agg(
             N="count",
             RMSE_mm=lambda x: float(np.sqrt(np.mean(np.square(x)))),
             MAE_mm="mean",
             Median_mm="median",
             P95_mm=lambda x: float(np.percentile(x, 95)),
         )
         .reset_index()
         .sort_values("RMSE_mm", ascending=False)
    )

    per_mouse = (
        m.groupby("mouse_id")["err_mm"]
         .agg(
             N="count",
             RMSE_mm=lambda x: float(np.sqrt(np.mean(np.square(x)))),
             MAE_mm="mean",
             Median_mm="median",
             P95_mm=lambda x: float(np.percentile(x, 95)),
         )
         .reset_index()
         .sort_values("mouse_id")
    )

    return {"overall": overall, "offsets": offsets, "per_node": per_node, "per_mouse": per_mouse, "merged": m}


# -----------------------------
# Reprojection diagnostics
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
    import json
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


def reprojection_stats(
    cams: List[Camera],
    coords_df: pd.DataFrame,
    obs_cam: np.ndarray, obs_mouse: np.ndarray, obs_joint: np.ndarray, obs_frame: np.ndarray,
    obs_x: np.ndarray, obs_y: np.ndarray, obs_l: np.ndarray,
    min_likelihood: float = 0.05,
) -> pd.DataFrame:
    """Compute reprojection error (px) vs DLC observations."""
    # Build lookup: (frame, mouse_id, joint_name) -> Pw
    # coords_df is dense; create array world[M,T,J,3] would be faster, but keep it simple & safe.
    key_to_xyz = {}
    for row in coords_df.itertuples(index=False):
        # columns: frame,time,mouse_id,behavior,node,x,y,z
        key_to_xyz[(int(row.frame), int(row.mouse_id), str(row.node))] = (float(row.x), float(row.y), float(row.z))

    cam_name_to_cam = {c.name: c for c in cams}
    # obs_cam is int indices (0..2) in frontend. Map to names via cams order.
    cam_index_to_name = {i: cams[i].name for i in range(len(cams))}

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
        Pw = np.array(key_to_xyz[key], dtype=np.float64)
        cam_name = cam_index_to_name[int(obs_cam[i])]
        cam = cam_name_to_cam[cam_name]
        u, v, vis = cam.project(Pw)
        if not vis:
            continue
        dx = u - float(obs_x[i])
        dy = v - float(obs_y[i])
        e = math.sqrt(dx*dx + dy*dy)
        errs.append((cam_name, e))
    if not errs:
        return pd.DataFrame([{ "N": 0, "RMSE_px": np.nan, "MAE_px": np.nan, "Median_px": np.nan, "P95_px": np.nan }])

    df = pd.DataFrame(errs, columns=["camera", "err_px"])
    overall = pd.DataFrame([{
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
    # return both as one multi-table-like DF list? we'll return overall with a marker and then per-cam
    overall.insert(0, "camera", "ALL")
    return pd.concat([overall, per_cam], ignore_index=True)


# -----------------------------
# Identity tracking without GT
# -----------------------------
def identity_track_two_mice(world: np.ndarray, anchor_nodes: Tuple[str, str] = ("head", "tail_root")) -> np.ndarray:
    """Return a swapped copy of world[M,T,J,3] to minimize identity switches over time.

    Uses dynamic programming for M=2. Cost compares anchor displacement frame-to-frame.
    """
    M, T, J, _ = world.shape
    if M != 2:
        return world

    anchor_idx = [JOINTS.index(n) for n in anchor_nodes if n in JOINTS]
    if not anchor_idx:
        return world

    # anchor position per mouse, frame: average of available anchors
    anchor = np.full((M, T, 3), np.nan, dtype=np.float64)
    for m in range(M):
        for t in range(T):
            pts = []
            for j in anchor_idx:
                pts.append(world[m, t, j])
            pts = np.asarray(pts)
            anchor[m, t] = pts.mean(axis=0)

    def dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return 1e6
        d = a - b
        return float(np.sqrt(np.dot(d, d)))

    # dp[t, state] where state 0 = no swap at t, state 1 = swapped at t
    dp = np.full((T, 2), np.inf)
    prev = np.full((T, 2), -1, dtype=int)

    # cost at first frame = 0
    dp[0, 0] = 0.0
    dp[0, 1] = 0.0

    for t in range(1, T):
        for state in (0, 1):
            # mapping at t: if state==0 -> (0->0,1->1), if state==1 -> (0->1,1->0)
            map_t = (0, 1) if state == 0 else (1, 0)
            for pst in (0, 1):
                map_prev = (0, 1) if pst == 0 else (1, 0)
                c = dist(anchor[map_t[0], t], anchor[map_prev[0], t-1]) + dist(anchor[map_t[1], t], anchor[map_prev[1], t-1])
                v = dp[t-1, pst] + c
                if v < dp[t, state]:
                    dp[t, state] = v
                    prev[t, state] = pst

    # backtrack best final state
    state = int(np.argmin(dp[T-1]))
    states = [state]
    for t in range(T-1, 0, -1):
        state = prev[t, state]
        states.append(state)
    states = list(reversed(states))

    out = world.copy()
    swapped_frames = 0
    for t, st in enumerate(states):
        if st == 1:
            out[:, t] = out[::-1, t]
            swapped_frames += 1

    print(f"[identity-track] swapped {swapped_frames}/{T} frames ({100.0*swapped_frames/T:.1f}%) to minimize anchor jumps.")
    return out


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

    # cage corners + edges
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
        d = {row["node"]: np.array([row["x"], row["y"], row["z"]], dtype=np.float64) for _, row in g.iterrows()}
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
# CLI + main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baWithPathModel_v5.py")
    p.add_argument("--cameras", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--dlc", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--out", required=True)
    p.add_argument("--gt", default=None)

    p.add_argument("--quat-order", choices=["auto", "wxyz", "xyzw"], default="auto")
    p.add_argument("--pose-mode", choices=["auto", "body_to_world", "world_to_body"], default="auto")
    p.add_argument("--allow-mouse-swap", action="store_true")

    p.add_argument("--init-likelihood-min", type=float, default=0.2)

    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--optimize-cameras", action="store_true")
    p.add_argument("--optimize-intrinsics", action="store_true")

    p.add_argument("--disable-path", action="store_true")
    p.add_argument("--rigid-only", action="store_true")

    p.add_argument("--identity-track", action="store_true",
                   help="Post-process identities to reduce switching (2 mice) using continuity of head+tail_root.")
    p.add_argument("--reproj-min-likelihood", type=float, default=0.05,
                   help="Min DLC likelihood for reprojection diagnostics.")

    p.add_argument("--write-video", action="store_true")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--video-labels", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    cam_quat, cam_trans, cam_intr = load_cameras_json(args.cameras)
    template_pts = load_template_points(args.template)

    (obs_cam, obs_mouse, obs_joint, obs_frame,
     obs_x, obs_y, obs_l, num_frames) = load_dlc_long_csv(args.dlc)

    init_q, init_t = init_poses_from_triangulation(
        template_pts,
        obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l,
        cam_quat, cam_trans, cam_intr,
        likelihood_min=args.init_likelihood_min,
    )

    wp = default_weight_pack().to_dict()
    wp["knot_interval_frames"] = max(3, int(round(args.fps * 0.3)))

    if args.disable_path:
        wp["sigma_spline_p_mm"] = 1e9
        wp["sigma_spline_R_rad"] = 1e9

    if args.rigid_only:
        tiny = 1e-6
        wp["sigma_delta_mm"] = {j: tiny for j in range(len(JOINTS))}
        wp["sigma_deltadot_mm_per_frame"] = {j: tiny for j in range(len(JOINTS))}

    solver_opts = {
        "max_num_iterations": int(args.max_iter),
        "num_threads": int(args.threads),
        "verbose": bool(args.verbose),
        "optimize_cameras": bool(args.optimize_cameras),
        "optimize_intrinsics": bool(args.optimize_intrinsics),
    }

    out = mouse_ba.solve(
        cam_quat, cam_trans, cam_intr,
        template_pts,
        obs_cam, obs_mouse, obs_joint, obs_frame,
        obs_x, obs_y, obs_l,
        int(num_frames),
        init_q, init_t,
        wp,
        solver_opts,
    )
    print(out["summary"])

    mouse_quat = np.asarray(out["mouse_quat"])
    mouse_trans = np.asarray(out["mouse_trans"])
    slack = np.asarray(out["slack"])

    # choose conventions using GT if requested
    chosen_order = args.quat_order if args.quat_order != "auto" else "wxyz"
    chosen_pose = args.pose_mode if args.pose_mode != "auto" else "body_to_world"
    chosen_swap = False

    def eval_config(order: str, pose: str, do_swap: bool):
        world = compute_world_points(template_pts, mouse_quat, mouse_trans, slack, order, pose)
        if do_swap and world.shape[0] == 2:
            world = world[[1, 0], ...]
        df = build_coords_df_from_world(world, fps=float(args.fps), behavior_default=0)
        st = compare_to_ground_truth(df, args.gt) if args.gt else None
        rmse = float(st["overall"]["RMSE_mm"].iloc[0]) if st is not None else float("nan")
        return rmse, st, df, world

    stats = None
    est_df = None
    world = None

    if args.gt and (args.quat_order == "auto" or args.pose_mode == "auto"):
        orders = ["wxyz", "xyzw"] if args.quat_order == "auto" else [args.quat_order]
        poses = ["body_to_world", "world_to_body"] if args.pose_mode == "auto" else [args.pose_mode]
        candidates = []
        for o in orders:
            for pz in poses:
                candidates.append((o, pz, False))
                if args.allow_mouse_swap:
                    candidates.append((o, pz, True))
        best = None
        for o, pz, sw in candidates:
            rmse, st, df, wld = eval_config(o, pz, sw)
            if best is None or rmse < best[0]:
                best = (rmse, o, pz, sw, st, df, wld)
        assert best is not None
        _, chosen_order, chosen_pose, chosen_swap, stats, est_df, world = best
        print(f"[diagnostic] Selected quat_order={chosen_order} pose_mode={chosen_pose} mouse_id_swap={chosen_swap} based on lowest RMSE.")
    else:
        world = compute_world_points(template_pts, mouse_quat, mouse_trans, slack, chosen_order, chosen_pose)
        if chosen_swap and world.shape[0] == 2:
            world = world[[1, 0], ...]
        est_df = build_coords_df_from_world(world, fps=float(args.fps), behavior_default=0)
        if args.gt:
            stats = compare_to_ground_truth(est_df, args.gt)

    # optional identity tracking (no GT needed)
    if args.identity_track:
        world = identity_track_two_mice(world)
        est_df = build_coords_df_from_world(world, fps=float(args.fps), behavior_default=0)
        if args.gt:
            stats = compare_to_ground_truth(est_df, args.gt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    est_df.to_csv(out_path, index=False)
    print(f"Wrote estimated 3D coordinates to: {out_path}")

    if args.gt and stats is not None:
        print("\n=== Accuracy vs ground truth (mm) ===")
        print(stats["overall"].to_string(index=False))
        print("\nMean axis offsets (est - gt) and std:")
        print(stats["offsets"].to_string(index=False))
        print("\nPer-mouse:")
        print(stats["per_mouse"].to_string(index=False))
        print("\nPer-node (worst first):")
        print(stats["per_node"].to_string(index=False))

    # reprojection diagnostics
    cams, _ = load_scene_cameras(args.cameras)
    rep_est = reprojection_stats(cams, est_df, obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l,
                                min_likelihood=float(args.reproj_min_likelihood))
    print("\n=== Reprojection error vs DLC (px) for EST ===")
    print(rep_est.to_string(index=False))

    if args.gt:
        gt_df = pd.read_csv(args.gt)
        rep_gt = reprojection_stats(cams, gt_df, obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l,
                                   min_likelihood=float(args.reproj_min_likelihood))
        print("\n=== Reprojection error vs DLC (px) for GT ===")
        print(rep_gt.to_string(index=False))
        print("(If GT reprojection error is large, the GT file likely does NOT correspond to this DLC/camera dataset.)")

    if args.write_video:
        video_dir = Path(args.video_dir) if args.video_dir else out_path.parent
        write_videos(args.cameras, args.template, est_df, video_dir, fps=float(args.fps), draw_labels=bool(args.video_labels))
        print(f"Wrote videos to: {video_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
