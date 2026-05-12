from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CageBBox:
    """Axis-aligned bounding box in world coordinates (mm)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class CameraModel:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # (3,3) world->cam
    t: np.ndarray  # (3,)  world->cam
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float)

    @property
    def C_world(self) -> np.ndarray:
        # Camera center in world coordinates: C = -R^T t
        return -self.R.T @ self.t

    def project(self, Pw: np.ndarray) -> Tuple[float, float]:
        """Project a world point Pw (3,) to undistorted pixel coordinates."""
        Pc = self.R @ Pw + self.t
        if Pc[2] <= 1e-12:
            return float('nan'), float('nan')
        u = self.fx * (Pc[0] / Pc[2]) + self.cx
        v = self.fy * (Pc[1] / Pc[2]) + self.cy
        return float(u), float(v)

    def ray_world(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (origin_world, dir_world_unit) for pixel uv (2,) undistorted."""
        u, v = float(uv[0]), float(uv[1])
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d_cam = np.array([x, y, 1.0], dtype=float)
        d_cam /= np.linalg.norm(d_cam) + 1e-15
        d_world = self.R.T @ d_cam
        d_world /= np.linalg.norm(d_world) + 1e-15
        return self.C_world.copy(), d_world


@dataclass
class CameraRig:
    cameras: List[CameraModel]
    name_to_index: Dict[str, int]
    cage_bbox: CageBBox

    def as_ceres_dict(self) -> Dict[str, np.ndarray]:
        """Return dict of arrays in the format expected by ceres_point_ba.solve_point."""
        C = len(self.cameras)
        fx = np.zeros((C,), dtype=float)
        fy = np.zeros((C,), dtype=float)
        cx = np.zeros((C,), dtype=float)
        cy = np.zeros((C,), dtype=float)
        R = np.zeros((C, 3, 3), dtype=float)
        t = np.zeros((C, 3), dtype=float)
        for i, cam in enumerate(self.cameras):
            fx[i], fy[i], cx[i], cy[i] = cam.fx, cam.fy, cam.cx, cam.cy
            R[i] = cam.R
            t[i] = cam.t
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "R": R, "t": t}


def load_cameras_json(path: str) -> CameraRig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cage = data.get("cage", {})
    origin = cage.get("origin", [0.0, 0.0, 0.0])
    ox, oy, oz = map(float, origin)
    width = float(cage.get("width", 0.0))
    depth = float(cage.get("depth", 0.0))
    height = float(cage.get("height", 0.0))

    bbox = CageBBox(
        x_min=ox,
        x_max=ox + width,
        y_min=oy,
        y_max=oy + depth,
        z_min=oz,
        z_max=oz + height,
    )

    cams = []
    name_to_idx: Dict[str, int] = {}

    for cam_entry in data["cameras"]:
        name = cam_entry["name"]
        cam = CameraModel(
            name=name,
            fx=float(cam_entry["fx"]),
            fy=float(cam_entry["fy"]),
            cx=float(cam_entry["cx"]),
            cy=float(cam_entry["cy"]),
            R=np.array(cam_entry["R"], dtype=float),
            t=np.array(cam_entry["t"], dtype=float),
            width=int(cam_entry.get("width", 0)),
            height=int(cam_entry.get("height", 0)),
        )
        name_to_idx[name] = len(cams)
        cams.append(cam)

    # Keep the input order from JSON (usually cam1_top, cam2_front, cam3_side)
    return CameraRig(cameras=cams, name_to_index=name_to_idx, cage_bbox=bbox)


def fundamental_matrix(cam1: CameraModel, cam2: CameraModel) -> np.ndarray:
    """Compute F such that x2^T F x1 = 0 for pixel homogeneous coords.

    Assumes world->cam extrinsics: Pc = R Pw + t.
    """
    K1 = cam1.K
    K2 = cam2.K

    R1, t1 = cam1.R, cam1.t.reshape(3, 1)
    R2, t2 = cam2.R, cam2.t.reshape(3, 1)

    # Relative transform from cam1 to cam2 (in cam2 coordinates)
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    tx = np.array(
        [[0.0, -t_rel[2, 0], t_rel[1, 0]], [t_rel[2, 0], 0.0, -t_rel[0, 0]], [-t_rel[1, 0], t_rel[0, 0], 0.0]],
        dtype=float,
    )

    E = tx @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F
