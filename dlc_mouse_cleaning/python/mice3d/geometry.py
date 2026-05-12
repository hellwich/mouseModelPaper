from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .cameras import CameraRig, fundamental_matrix
from .dlc import Observation


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3,) unit


def build_ray(rig: CameraRig, obs: Observation) -> Ray:
    cam = rig.cameras[obs.cam_idx]
    o, d = cam.ray_world(obs.uv)
    return Ray(origin=o, direction=d)


def ray_angle_rad(r1: Ray, r2: Ray) -> float:
    c = float(np.clip(np.dot(r1.direction, r2.direction), -1.0, 1.0))
    # using absolute dot makes angle symmetric w.r.t. direction sign
    c = abs(c)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def closest_points_on_rays(r1: Ray, r2: Ray) -> Tuple[np.ndarray, np.ndarray]:
    """Return closest points P1 on r1 and P2 on r2."""
    p1, d1 = r1.origin, r1.direction
    p2, d2 = r2.origin, r2.direction

    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-12:
        # nearly parallel: pick arbitrary
        s = 0.0
        t = e / c if c > 1e-12 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

    P1 = p1 + s * d1
    P2 = p2 + t * d2
    return P1, P2


def ray_pair_min_distance(r1: Ray, r2: Ray) -> float:
    P1, P2 = closest_points_on_rays(r1, r2)
    return float(np.linalg.norm(P1 - P2))


def pair_midpoint_init(rig: CameraRig, obs_a: Observation, obs_b: Observation) -> np.ndarray:
    r1 = build_ray(rig, obs_a)
    r2 = build_ray(rig, obs_b)
    P1, P2 = closest_points_on_rays(r1, r2)
    return 0.5 * (P1 + P2)


def bbox_gate(Pw: np.ndarray, rig: CameraRig, margin_mm: float = 0.0) -> bool:
    b = rig.cage_bbox
    x, y, z = float(Pw[0]), float(Pw[1]), float(Pw[2])
    return (
        (b.x_min - margin_mm) <= x <= (b.x_max + margin_mm)
        and (b.y_min - margin_mm) <= y <= (b.y_max + margin_mm)
        and (b.z_min - margin_mm) <= z <= (b.z_max + margin_mm)
    )


def fast_reprojection_errors_px(rig: CameraRig, Pw: np.ndarray, obs_list: List[Observation]) -> np.ndarray:
    errs = []
    for obs in obs_list:
        cam = rig.cameras[obs.cam_idx]
        u_hat, v_hat = cam.project(Pw)
        if not np.isfinite(u_hat) or not np.isfinite(v_hat):
            errs.append(1e6)
            continue
        du = u_hat - float(obs.uv[0])
        dv = v_hat - float(obs.uv[1])
        errs.append(np.hypot(du, dv))
    return np.array(errs, dtype=float)


def epipolar_distance_px(rig: CameraRig, obs1: Observation, obs2: Observation, F12: np.ndarray | None = None) -> float:
    """Distance of obs2 to epipolar line induced by obs1: l2 = F x1."""
    cam1 = rig.cameras[obs1.cam_idx]
    cam2 = rig.cameras[obs2.cam_idx]
    if F12 is None:
        F12 = fundamental_matrix(cam1, cam2)

    x1 = np.array([obs1.uv[0], obs1.uv[1], 1.0], dtype=float)
    x2 = np.array([obs2.uv[0], obs2.uv[1], 1.0], dtype=float)
    l2 = F12 @ x1

    a, b, c = float(l2[0]), float(l2[1]), float(l2[2])
    denom = np.hypot(a, b)
    if denom < 1e-12:
        return float('inf')
    return abs(a * x2[0] + b * x2[1] + c) / denom


def min_pairwise_ray_angle_rad(rig: CameraRig, obs_list: List[Observation]) -> float:
    rays = [build_ray(rig, o) for o in obs_list]
    m = float('inf')
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            m = min(m, ray_angle_rad(rays[i], rays[j]))
    return m


def max_angle_pair(rig: CameraRig, obs_list: List[Observation]) -> Tuple[int, int]:
    rays = [build_ray(rig, o) for o in obs_list]
    best = (-1.0, (0, 1))
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            ang = ray_angle_rad(rays[i], rays[j])
            if ang > best[0]:
                best = (ang, (i, j))
    return best[1]


def distance_point_to_line(P: np.ndarray, line_point: np.ndarray, line_dir_unit: np.ndarray) -> float:
    v = P - line_point
    proj = np.dot(v, line_dir_unit)
    perp = v - proj * line_dir_unit
    return float(np.linalg.norm(perp))
