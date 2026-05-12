import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


JOINTS = [
    "head",
    "nose_tip",
    "left_ear_tip",
    "right_ear_tip",
    "tail_root",
    "tail_tip",
    "left_front_paw",
    "right_front_paw",
    "left_hind_paw",
    "right_hind_paw",
]

JOINT_TO_IDX = {n: i for i, n in enumerate(JOINTS)}
CAM_ORDER = ["cam1_top", "cam2_front", "cam3_side"]
CAM_TO_IDX = {n: i for i, n in enumerate(CAM_ORDER)}
MOUSE_TO_IDX = {"mouse0": 0, "mouse1": 1}


@dataclass
class WeightPack:
    likelihood_min: float = 0.05
    sigma_pix_min: float = 1.5
    sigma_pix_max: float = 8.0
    robust_loss: str = "huber"  # huber|cauchy|none
    robust_param_px: float = 3.0

    knot_interval_frames: int = 3  # 10 FPS -> 0.3 s
    sigma_spline_p_mm: float = 4.0
    sigma_spline_R_rad: float = np.deg2rad(5.0)

    # Joint weights (index -> weight)
    joint_weight: Dict[int, float] = None

    # Slack priors and smoothness
    sigma_delta_mm: Dict[int, float] = None
    sigma_deltadot_mm_per_frame: Dict[int, float] = None

    def to_dict(self) -> dict:
        return {
            "likelihood_min": self.likelihood_min,
            "sigma_pix_min": self.sigma_pix_min,
            "sigma_pix_max": self.sigma_pix_max,
            "robust_loss": self.robust_loss,
            "robust_param_px": self.robust_param_px,
            "knot_interval_frames": self.knot_interval_frames,
            "sigma_spline_p_mm": self.sigma_spline_p_mm,
            "sigma_spline_R_rad": float(self.sigma_spline_R_rad),
            "joint_weight": self.joint_weight,
            "sigma_delta_mm": self.sigma_delta_mm,
            "sigma_deltadot_mm_per_frame": self.sigma_deltadot_mm_per_frame,
        }


def default_weight_pack() -> WeightPack:
    wp = WeightPack()

    # Joint weights
    jw = {i: 0.3 for i in range(len(JOINTS))}
    jw[JOINT_TO_IDX["head"]] = 1.0
    jw[JOINT_TO_IDX["tail_root"]] = 1.0
    jw[JOINT_TO_IDX["left_ear_tip"]] = 0.5
    jw[JOINT_TO_IDX["right_ear_tip"]] = 0.5
    jw[JOINT_TO_IDX["tail_tip"]] = 0.1
    wp.joint_weight = jw

    # Slack priors (mm)
    sd = {}
    sd[JOINT_TO_IDX["left_ear_tip"]] = 1.5
    sd[JOINT_TO_IDX["right_ear_tip"]] = 1.5
    sd[JOINT_TO_IDX["nose_tip"]] = 2.0
    sd[JOINT_TO_IDX["left_front_paw"]] = 2.5
    sd[JOINT_TO_IDX["right_front_paw"]] = 2.5
    sd[JOINT_TO_IDX["left_hind_paw"]] = 3.0
    sd[JOINT_TO_IDX["right_hind_paw"]] = 3.0
    sd[JOINT_TO_IDX["tail_tip"]] = 10.0
    wp.sigma_delta_mm = sd

    # Slack smoothness (mm/frame)
    sdd = {}
    sdd[JOINT_TO_IDX["left_ear_tip"]] = 0.8
    sdd[JOINT_TO_IDX["right_ear_tip"]] = 0.8
    sdd[JOINT_TO_IDX["nose_tip"]] = 1.0
    sdd[JOINT_TO_IDX["left_front_paw"]] = 1.2
    sdd[JOINT_TO_IDX["right_front_paw"]] = 1.2
    sdd[JOINT_TO_IDX["left_hind_paw"]] = 1.2
    sdd[JOINT_TO_IDX["right_hind_paw"]] = 1.2
    sdd[JOINT_TO_IDX["tail_tip"]] = 3.0
    wp.sigma_deltadot_mm_per_frame = sdd

    return wp


def load_cameras_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    cams = {c["name"]: c for c in data["cameras"]}

    cam_quat = np.zeros((3, 4), dtype=np.float64)
    cam_trans = np.zeros((3, 3), dtype=np.float64)
    cam_intr = np.zeros((3, 4), dtype=np.float64)

    for i, name in enumerate(CAM_ORDER):
        c = cams[name]
        R = np.array(c["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(c["t"], dtype=np.float64).reshape(3)
        fx, fy, cx, cy = float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])
        cam_intr[i] = [fx, fy, cx, cy]
        cam_trans[i] = t
        cam_quat[i] = rotmat_to_quat_wxyz(R)

    return cam_quat, cam_trans, cam_intr


def load_template_points(path: str):
    pts = np.zeros((len(JOINTS), 3), dtype=np.float64)
    with open(path, "r") as f:
        mode = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "[NODES]":
                mode = "nodes"; continue
            if line == "[EDGES]":
                break
            if mode == "nodes":
                parts = line.split()
                name = parts[0]
                if name not in JOINT_TO_IDX:
                    continue
                idx = JOINT_TO_IDX[name]
                pts[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return pts


def load_dlc_long_csv(path: str):
    df = pd.read_csv(path)

    # Normalize naming
    df = df[df["camera"].isin(CAM_TO_IDX)]
    df = df[df["individual"].isin(MOUSE_TO_IDX)]
    df = df[df["bodypart"].isin(JOINT_TO_IDX)]

    # Build arrays
    obs_cam = df["camera"].map(CAM_TO_IDX).astype(np.int32).to_numpy()
    obs_mouse = df["individual"].map(MOUSE_TO_IDX).astype(np.int32).to_numpy()
    obs_joint = df["bodypart"].map(JOINT_TO_IDX).astype(np.int32).to_numpy()
    obs_frame = df["frame"].astype(np.int32).to_numpy()

    obs_x = df["x"].astype(np.float64).to_numpy()
    obs_y = df["y"].astype(np.float64).to_numpy()
    obs_l = df["likelihood"].astype(np.float64).to_numpy()

    num_frames = int(df["frame"].max()) + 1
    return obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l, num_frames


def build_projection_matrices(cam_quat, cam_trans, cam_intr):
    P = []
    for i in range(cam_quat.shape[0]):
        R = quat_to_rotmat_wxyz(cam_quat[i])
        t = cam_trans[i].reshape(3, 1)
        fx, fy, cx, cy = cam_intr[i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        Rt = np.hstack([R, t])
        P.append(K @ Rt)
    return P


def triangulate_point(P_list: List[np.ndarray], xy_list: List[Tuple[float, float]]):
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


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    # src, dst: (N,3) with correspondence; returns R (3,3), t (3,)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    S = X.T @ Y / src.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def init_poses_from_triangulation(
    template_pts: np.ndarray,
    obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l,
    cam_quat, cam_trans, cam_intr,
    likelihood_min: float = 0.2,
):
    Pmats = build_projection_matrices(cam_quat, cam_trans, cam_intr)

    M = 2
    T = int(obs_frame.max()) + 1
    init_q = np.zeros((M, T, 4), dtype=np.float64)
    init_t = np.zeros((M, T, 3), dtype=np.float64)

    # Identity init
    init_q[..., 0] = 1.0

    # Build index: (m,t,j,c) -> (x,y,l)
    # For speed, pre-bucket rows by (m,t,j)
    buckets = {}
    for c, m, j, t, x, y, l in zip(obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if l < likelihood_min:
            continue
        key = (int(m), int(t), int(j))
        buckets.setdefault(key, []).append((int(c), float(x), float(y)))

    # For each (m,t), triangulate all joints with >=2 views
    for m in range(M):
        last_R = np.eye(3)
        last_t = np.zeros(3)
        for t in range(T):
            src_pts = []
            dst_pts = []
            for j in range(template_pts.shape[0]):
                key = (m, t, j)
                if key not in buckets:
                    continue
                obs = buckets[key]
                if len(obs) < 2:
                    continue
                # Use all available cameras for this joint
                P_list = [Pmats[o[0]] for o in obs]
                xy_list = [(o[1], o[2]) for o in obs]
                X = triangulate_point(P_list, xy_list)
                src_pts.append(template_pts[j])
                dst_pts.append(X)

            if len(src_pts) >= 3:
                R, tt = umeyama_alignment(np.stack(src_pts), np.stack(dst_pts))
                last_R, last_t = R, tt
            # else keep last
            init_q[m, t] = rotmat_to_quat_wxyz(last_R)
            init_t[m, t] = last_t

    return init_q, init_t


def rotmat_to_quat_wxyz(R: np.ndarray):
    # Standard conversion, returns [w,x,y,z]
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return q


def quat_to_rotmat_wxyz(q: np.ndarray):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R
