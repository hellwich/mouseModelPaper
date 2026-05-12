#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import colorsys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy.interpolate import RBFInterpolator

from PIL import Image

import trimesh
import pyrender

import copy

import datetime
import shutil
from pathlib import Path

import pandas as pd  # type: ignore
import yaml  # type: ignore

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("OpenCV is required for writing MP4. Install opencv from conda-forge.") from e


def norm_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def parse_bgr(s: str) -> Tuple[int, int, int]:
    """Parse 'B,G,R' into a 3-tuple of ints in [0,255]."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected B,G,R (3 comma-separated ints), got: {s!r}")
    try:
        b, g, r = (int(p) for p in parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Bad B,G,R value: {s!r}") from e
    for v in (b, g, r):
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError(f"B,G,R values must be 0..255, got: {s!r}")
    return (b, g, r)


def _auto_color_bgr_for_id(mid: int) -> Tuple[int, int, int]:
    """Deterministic vivid color for mouse IDs other than 0/1 (fallback)."""
    # Golden-ratio hue stepping for good separation.
    h = (mid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def mouse_id_to_bgr(mid: int, mouse0_bgr: Tuple[int, int, int], mouse1_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if mid == 0:
        return mouse0_bgr
    if mid == 1:
        return mouse1_bgr
    return _auto_color_bgr_for_id(mid)



# ----------------------------
# IO: mesh keypoints
# ----------------------------
def load_nodes_txt(path: str) -> Dict[str, np.ndarray]:
    """
    Reads:
      [NODES]
      head  x y z
      ...

    Ignores [EDGES] and anything after.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    nodes: Dict[str, np.ndarray] = {}
    mode = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper() == "[NODES]":
                mode = "nodes"
                continue
            if line.upper() == "[EDGES]":
                break
            if mode == "nodes":
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Bad node line: {line}")
                name = norm_label(parts[0])
                x, y, z = map(float, parts[1:])
                nodes[name] = np.array([x, y, z], dtype=np.float64)

    if not nodes:
        raise ValueError(f"No nodes found in {path}")
    return nodes


# ----------------------------
# IO: Mouse 1 coords_3d.csv (frame-indexed)
# ----------------------------
@dataclass
class MouseFrameData:
    behavior: int
    pts: Dict[str, np.ndarray]  # label -> (3,)


@dataclass
class FrameData:
    time: float
    mice: Dict[int, MouseFrameData]

def load_coords_3d_csv(path: str) -> Dict[int, FrameData]:
    """
    Reads coords_3d.csv produced by mouse_sim.py (single mouse) or mouse_sim2.py (multi mouse).

    Required columns:
      frame,time,behavior,node,x,y,z
    Optional:
      mouse_id
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    frames: Dict[int, FrameData] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"frame", "time", "behavior", "node", "x", "y", "z"}
        fieldnames = set(r.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(f"{path} missing required columns. Have: {r.fieldnames}")

        has_mouse_id = "mouse_id" in fieldnames

        for row in r:
            fi = int(row["frame"])
            t = float(row["time"])
            b = int(row["behavior"])
            mid = int(row["mouse_id"]) if has_mouse_id and row.get("mouse_id", "") != "" else 0

            node = norm_label(row["node"])
            p = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)

            if fi not in frames:
                frames[fi] = FrameData(time=t, mice={})
            if mid not in frames[fi].mice:
                frames[fi].mice[mid] = MouseFrameData(behavior=b, pts={})

            frames[fi].mice[mid].pts[node] = p
            frames[fi].mice[mid].behavior = b

    if not frames:
        raise ValueError(f"No frames loaded from {path}")
    return frames


# ----------------------------
# IO: cameras.json from mouse-sim
# ----------------------------
@dataclass
class CameraSpec:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: np.ndarray  # world -> cam
    t_wc: np.ndarray  # world -> cam


def load_cameras_json(path: str) -> List[CameraSpec]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cams = []
    for c in data["cameras"]:
        cams.append(CameraSpec(
            name=c["name"],
            width=int(c["width"]),
            height=int(c["height"]),
            fx=float(c["fx"]),
            fy=float(c["fy"]),
            cx=float(c["cx"]),
            cy=float(c["cy"]),
            R_wc=np.array(c["R"], dtype=np.float64),
            t_wc=np.array(c["t"], dtype=np.float64),
        ))
    if not cams:
        raise ValueError("No cameras in cameras.json")
    return cams


def load_scene_json(path: str):
    """
    Supports both:
      (new) {"cage": {...}, "cameras": [...]}
      (old) {"cameras": [...]}   (no cage)
    Returns: (cage_dict_or_None, list[CameraSpec])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cage = data.get("cage", None)

    cam_list = data.get("cameras", None)
    if cam_list is None:
        raise ValueError("JSON must contain key 'cameras'")

    cams: List[CameraSpec] = []
    for c in cam_list:
        cams.append(CameraSpec(
            name=c["name"],
            width=int(c["width"]),
            height=int(c["height"]),
            fx=float(c["fx"]),
            fy=float(c["fy"]),
            cx=float(c["cx"]),
            cy=float(c["cy"]),
            R_wc=np.array(c["R"], dtype=np.float64),
            t_wc=np.array(c["t"], dtype=np.float64),
        ))

    if not cams:
        raise ValueError("No cameras found in JSON")

    return cage, cams


def world_to_cam_to_pose(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """
    Convert world->cam extrinsics in OpenCV convention (x right, y down, z forward)
    to an OpenGL/pyrender camera pose (cam->world, looking down -Z, y up).
    """
    # CV -> GL axis flip (keeps x, flips y and z)
    cv_to_gl = np.diag([1.0, -1.0, -1.0])

    R_wc_gl = cv_to_gl @ R_wc
    t_wc_gl = cv_to_gl @ t_wc

    # invert world->cam to cam->world pose
    R_cw = R_wc_gl.T
    t_cw = -R_cw @ t_wc_gl

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_cw
    pose[:3, 3] = t_cw
    return pose

def _load_img(path: str | None):
    if not path:
        return None
    img = Image.open(path).convert("RGBA")
    return img


def swap_rb_image(img: Image.Image) -> Image.Image:
    """Return a copy with R and B channels swapped (BGR-as-RGB)."""
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    arr = arr[..., [2, 1, 0, 3]]
    return Image.fromarray(arr, mode="RGBA")


def make_swapped_visual(visual):
    """
    Deep-copy a trimesh visual and swap R/B channels for its texture image or vertex colors.
    This is used to make a selected mouse visually distinct without changing geometry.
    """
    if visual is None:
        return None
    v = copy.deepcopy(visual)

    # Texture-based visuals: try material.image first
    try:
        mat = getattr(v, "material", None)
        if mat is not None:
            img = getattr(mat, "image", None)
            if img is not None:
                mat.image = swap_rb_image(img)
    except Exception:
        pass

    # Some visuals store image directly
    try:
        img2 = getattr(v, "image", None)
        if img2 is not None:
            v.image = swap_rb_image(img2)
    except Exception:
        pass

    # Vertex colors
    try:
        vc = getattr(v, "vertex_colors", None)
        if vc is not None:
            vc = np.array(vc, copy=True)
            if vc.ndim == 2 and vc.shape[1] >= 3:
                vc[:, [0, 2]] = vc[:, [2, 0]]
            v.vertex_colors = vc
    except Exception:
        pass

    return v


def make_textured_quad(p00, p10, p11, p01, img_rgba, alpha=1.0):
    vertices = np.array([p00, p10, p11, p01], dtype=np.float64)

    # Two-sided: add reversed faces too
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [2, 1, 0], [3, 2, 0],
    ], dtype=np.int64)

    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)

    if img_rgba is None:
        color = np.array([255, 255, 255, 255], dtype=np.uint8)
        visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, (4, 1)))
        m = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        return pyrender.Mesh.from_trimesh(m, smooth=False)

    if alpha < 1.0:
        arr = np.array(img_rgba, dtype=np.uint8)
        arr[..., 3] = (arr[..., 3].astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        img_rgba = Image.fromarray(arr, mode="RGBA")

    material = trimesh.visual.material.SimpleMaterial(image=img_rgba)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img_rgba, material=material)
    m = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return pyrender.Mesh.from_trimesh(m, smooth=False)

def camera_center_world(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    return -R_wc.T @ t_wc

def camera_forward_world_cv(R_wc: np.ndarray) -> np.ndarray:
    # In CV convention, camera looks along +Z in camera coords.
    return R_wc.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

def pick_face_to_skip(C: np.ndarray, d: np.ndarray, cage: dict, eps: float = 1e-9) -> str | None:
    """
    Returns one of: 'T','B','N','S','E','W' to skip, or None.
    Cage coords: x in [0,w], y in [0,dep], z in [0,h]
    """
    w = float(cage["width"])
    dep = float(cage["depth"])
    h = float(cage["height"])

    best = (np.inf, None)

    def consider(face: str, s: float, P: np.ndarray):
        nonlocal best
        if s <= eps:
            return
        x, y, z = P
        if face in ("E", "W"):
            if (0 - eps) <= y <= (dep + eps) and (0 - eps) <= z <= (h + eps):
                best = min(best, (s, face))
        elif face in ("N", "S"):
            if (0 - eps) <= x <= (w + eps) and (0 - eps) <= z <= (h + eps):
                best = min(best, (s, face))
        elif face in ("T", "B"):
            if (0 - eps) <= x <= (w + eps) and (0 - eps) <= y <= (dep + eps):
                best = min(best, (s, face))

    # planes: x=0(W), x=w(E), y=0(S), y=dep(N), z=0(B), z=h(T)
    if abs(d[0]) > eps:
        s = (0.0 - C[0]) / d[0]
        consider("W", s, C + s * d)
        s = (w - C[0]) / d[0]
        consider("E", s, C + s * d)

    if abs(d[1]) > eps:
        s = (0.0 - C[1]) / d[1]
        consider("S", s, C + s * d)
        s = (dep - C[1]) / d[1]
        consider("N", s, C + s * d)

    if abs(d[2]) > eps:
        s = (0.0 - C[2]) / d[2]
        consider("B", s, C + s * d)
        s = (h - C[2]) / d[2]
        consider("T", s, C + s * d)

    return best[1]

def add_cage_to_scene(scene: pyrender.Scene, cage: dict, tex: dict, alpha: float, skip_face: str | None):
    w = float(cage["width"]); d = float(cage["depth"]); h = float(cage["height"])
    imgs = {k: _load_img(tex.get(k)) for k in ["T","B","N","S","E","W"]}

    def add(face, p00, p10, p11, p01):
        if skip_face == face:
            return
        scene.add(make_textured_quad(p00, p10, p11, p01, imgs[face], alpha))

    # Define each face as a quad in cage coords (UV mapping happens in make_textured_quad)
    add("B", [0,0,0], [w,0,0], [w,d,0], [0,d,0])
    add("T", [0,0,h], [w,0,h], [w,d,h], [0,d,h])

    add("S", [0,0,0], [w,0,0], [w,0,h], [0,0,h])
    add("N", [0,d,0], [w,d,0], [w,d,h], [0,d,h])

    add("W", [0,0,0], [0,d,0], [0,d,h], [0,0,h])
    add("E", [w,0,0], [w,d,0], [w,d,h], [w,0,h])

# ----------------------------
# Math: Kabsch rigid alignment
# ----------------------------
def kabsch(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds R,t minimizing ||R X + t - Y|| (least squares), with det(R)=+1
    X, Y: (N,3)
    Returns R(3,3), t(3,)
    """
    if X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError("X and Y must be (N,3) and same shape")

    x0 = X.mean(axis=0)
    y0 = Y.mean(axis=0)
    Xc = X - x0
    Yc = Y - y0
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = y0 - R @ x0
    return R, t


# ----------------------------
# Mesh loading (OBJ/PLY)
# ----------------------------
def load_meshes(mesh_path: str) -> List[trimesh.Trimesh]:
    """
    Returns a list of trimesh.Trimesh objects.
    Supports OBJ (with MTL+textures) and PLY.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)

    loaded = trimesh.load(mesh_path, force=None, process=False)

    meshes: List[trimesh.Trimesh] = []

    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        # Extract geometry. Keep each submesh (better chance to preserve visuals).
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
    else:
        raise ValueError(f"Unsupported mesh load result type: {type(loaded)}")

    if not meshes:
        raise ValueError(f"No meshes extracted from {mesh_path}")

    # sanity
    for m in meshes:
        if m.vertices is None or len(m.vertices) == 0 or m.faces is None or len(m.faces) == 0:
            raise ValueError("Loaded a mesh with no vertices/faces.")
    return meshes


def mesh_bbox_anchors(meshes: List[trimesh.Trimesh], scale: float = 1.2) -> np.ndarray:
    """
    Returns 8 bbox corners in mesh space, expanded by 'scale' around center.
    Used as residual anchors with zero displacement to stabilize RBF extrapolation.
    """
    all_bounds = np.array([m.bounds for m in meshes], dtype=np.float64)  # (K,2,3)
    bmin = all_bounds[:, 0, :].min(axis=0)
    bmax = all_bounds[:, 1, :].max(axis=0)
    c = 0.5 * (bmin + bmax)
    e = 0.5 * (bmax - bmin) * scale

    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(c + np.array([sx, sy, sz]) * e)
    return np.stack(corners, axis=0)  # (8,3)


# ----------------------------
# Deformation: rigid base + exact RBF residual
# ----------------------------
def deform_vertices_rigid_rbf(
    V_mesh: np.ndarray,                 # (Nv,3) mesh vertices
    Xk_mesh: np.ndarray,                # (Nk,3) keypoints in mesh coords (ALL constraints)
    Yk_world: np.ndarray,               # (Nk,3) target keypoints in world (frame)
    Xfit_mesh: np.ndarray,              # (Nf,3) fit subset in mesh coords
    Yfit_world: np.ndarray,             # (Nf,3) fit subset in world coords
    anchors_mesh: np.ndarray | None,    # (Na,3) in mesh coords
    kernel: str,
) -> np.ndarray:
    """
    Returns V_world_deformed (Nv,3).
    Enforces exact constraints at ALL keypoints via interpolatory RBF on residuals.
    """
    # rigid base (for numerical stability + nice global motion)
    R, t = kabsch(Xfit_mesh, Yfit_world)

    V_base = (V_mesh @ R.T) + t  # (Nv,3)
    Xk_base = (Xk_mesh @ R.T) + t
    residuals = Yk_world - Xk_base

    if anchors_mesh is not None and len(anchors_mesh) > 0:
        A_base = (anchors_mesh @ R.T) + t
        P = np.vstack([Xk_base, A_base])
        D = np.vstack([residuals, np.zeros((A_base.shape[0], 3), dtype=np.float64)])
    else:
        P = Xk_base
        D = residuals

    # Exact interpolation: smoothing=0
    rbf = RBFInterpolator(P, D, kernel=kernel, smoothing=0.0)

    V_def = V_base + rbf(V_base)
    return V_def


# ----------------------------
# Rendering
# ----------------------------
def make_scene_for_camera(cam: CameraSpec,
                          bg_color=(0, 0, 0),
                          ambient=(0.25, 0.25, 0.25)) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=list(bg_color), ambient_light=list(ambient))

    pyr_cam = pyrender.IntrinsicsCamera(cam.fx, cam.fy, cam.cx, cam.cy, znear=1e-3, zfar=1e6)
    cam_pose = world_to_cam_to_pose(cam.R_wc, cam.t_wc)
    scene.add(pyr_cam, pose=cam_pose)

    # lighting: directional + point tied to camera pose (so object is always lit)
    dir_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(dir_light, pose=cam_pose)

    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=cam_pose)

    return scene


def make_renderer_for_camera(cam: CameraSpec) -> pyrender.OffscreenRenderer:
    return pyrender.OffscreenRenderer(viewport_width=cam.width, viewport_height=cam.height)


def trimesh_to_pyrender(mesh: trimesh.Trimesh) -> pyrender.Mesh:
    # smooth=True gives nicer shading; textured visuals will pass through when supported
    return pyrender.Mesh.from_trimesh(mesh, smooth=True)

#from pyrender import RenderFlags


# ----------------------------
# DeepLabCut (DLC) project helpers
# ----------------------------
def _posix_relpath(path: str, start: str) -> str:
    # DLC stores frame paths with forward slashes, even on Windows.
    return Path(os.path.relpath(path, start)).as_posix()

def _project_point_cv(cam: CameraSpec, Pw: np.ndarray) -> Tuple[float, float, float]:
    """Project a world point to image plane using OpenCV-style intrinsics/extrinsics.
    Returns (u, v, z_cam) where z_cam is depth in camera (forward) direction."""
    Pc = cam.R_wc @ Pw + cam.t_wc
    z = float(Pc[2])
    if z <= 1e-9:
        return float('nan'), float('nan'), z
    u = cam.fx * float(Pc[0]) / z + cam.cx
    v = cam.fy * float(Pc[1]) / z + cam.cy
    return u, v, z

def _is_visible_from_depth(
    u: float,
    v: float,
    z: float,
    depth: np.ndarray,
    front_tol: float,
    patch_radius: int = 1,
    back_tol: Optional[float] = None,
) -> bool:
    """Patch-based, one-sided visibility test via depth buffer.

    A point is considered *occluded* only when rendered geometry is significantly
    closer to the camera than the point itself.

    Parameters
    ----------
    front_tol:
        Tolerance for *front* occluders. If d < z - front_tol, that pixel is in front
        of the point (occluding candidate). Larger values make the test less strict.
    patch_radius:
        Neighborhood radius in pixels around the projected point. 0 means single-pixel;
        1 means 3x3 patch. Small patches help thin/subpixel structures (tail tip, paws).
    back_tol:
        Optional upper tolerance to require d <= z + back_tol. Default None disables
        this check (recommended for thin structures/background-visible cases).
    """
    if not np.isfinite(u) or not np.isfinite(v) or z <= 0:
        return False
    H, W = depth.shape[:2]
    x = int(round(u))
    y = int(round(v))
    r = max(0, int(patch_radius))
    if x < -r or x >= W + r or y < -r or y >= H + r:
        return False

    x0 = max(0, x - r); x1 = min(W, x + r + 1)
    y0 = max(0, y - r); y1 = min(H, y + r + 1)
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    if not np.any(valid):
        return False
    dvals = patch[valid].astype(np.float32, copy=False)

    # One-sided occlusion logic:
    # visible if there exists at least one nearby rendered depth sample that is NOT in
    # front of the point by more than front_tol (and optionally not absurdly behind).
    if back_tol is None:
        return bool(np.any(dvals >= (z - front_tol)))
    return bool(np.any((dvals >= (z - front_tol)) & (dvals <= (z + back_tol))))

def _draw_dlc_overlay_markers(
    bgr: np.ndarray,
    cam: CameraSpec,
    fr: FrameData,
    mouse_ids: List[int],
    bodyparts: List[str],
    depth_map: np.ndarray,
    front_tol: float,
    vis_patch_radius: int = 1,
    back_tol: Optional[float] = None,
    visible_radius: int = 4,
    occluded_radius: int = 5,
) -> np.ndarray:
    """Draw visible points as filled blobs and occluded points as hollow circles.

    Visibility uses the provided depth map (typically mouse-only depth in --mouse-on-top mode).
    """
    out = bgr.copy()
    palette = [
        (0, 255, 0),    # green
        (0, 165, 255),  # orange
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan-ish in BGR
    ]
    H, W = out.shape[:2]
    for i, mid in enumerate(mouse_ids):
        mfr = fr.mice.get(mid)
        if mfr is None:
            continue
        color = palette[i % len(palette)]
        for bp in bodyparts:
            Pw = mfr.pts.get(bp)
            if Pw is None:
                continue
            u, v, z = _project_point_cv(cam, Pw)
            if not (np.isfinite(u) and np.isfinite(v) and z > 0):
                continue
            x = int(round(u)); y = int(round(v))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            vis = _is_visible_from_depth(u, v, z, depth_map, front_tol=front_tol, patch_radius=vis_patch_radius, back_tol=back_tol)
            if vis:
                cv2.circle(out, (x, y), visible_radius, color, thickness=-1, lineType=cv2.LINE_AA)
                # thin black outline to keep blobs visible on bright fur/background
                cv2.circle(out, (x, y), visible_radius, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(out, (x, y), occluded_radius, color, thickness=1, lineType=cv2.LINE_AA)
    return out

def _make_dlc_config(
    project_dir: str,
    task: str,
    scorer: str,
    date_str: str,
    individuals: List[str],
    bodyparts: List[str],
    cams: List[CameraSpec],
    video_paths: Dict[str, str],
) -> Dict:
    """Create a DLC multi-animal config dictionary (enough for DLC to pick up the project)."""
    project_path = os.path.abspath(project_dir)
    cfg: Dict = {
        'Task': task,
        'scorer': scorer,
        'date': date_str,
        'multianimalproject': True,
        'identity': False,
        'project_path': project_path,
        'video_sets': {},
        # In DLC multi-animal projects, 'bodyparts' is set to the sentinel 'MULTI!'
        # and actual per-individual bodyparts live in 'multianimalbodyparts'.
        'bodyparts': 'MULTI!',
        'multianimalbodyparts': bodyparts,
        'uniquebodyparts': [],
        'individuals': individuals,
        'skeleton': [],
        'skeleton_color': 'black',
        # common defaults (DLC will overwrite/add during dataset creation)
        'numframes2pick': 20,
        'TrainingFraction': [0.95],
        'iteration': 0,
        'default_net_type': 'dlcrnet_ms5',
        'default_augmenter': 'multi-animal-imgaug',
        'default_track_method': 'ellipse',
        'snapshotindex': -1,
        'pcutoff': 0.6,
        'dotsize': 8,
        'alphavalue': 0.7,
        'colormap': 'jet',
    }
    for cam in cams:
        vpath = os.path.abspath(video_paths[cam.name])
        cfg['video_sets'][vpath] = {
            'crop': f"0, {cam.width}, 0, {cam.height}",
        }
    return cfg

def _write_dlc_project(
    project_dir: str,
    cfg: Dict,
    overwrite: bool,
) -> None:
    os.makedirs(project_dir, exist_ok=True)
    # folders expected by DLC
    os.makedirs(os.path.join(project_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'labeled-data'), exist_ok=True)
    config_path = os.path.join(project_dir, 'config.yaml')
    if os.path.exists(config_path) and not overwrite:
        return
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def _write_dlc_collected_data(
    out_csv: str,
    out_h5: str,
    scorer: str,
    individuals: List[str],
    bodyparts: List[str],
    index_paths: List[str],
    rows_xy: List[List[float]],
) -> None:
    tuples = []
    for ind in individuals:
        for bp in bodyparts:
            tuples.append((scorer, ind, bp, 'x'))
            tuples.append((scorer, ind, bp, 'y'))
    cols = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    df = pd.DataFrame(rows_xy, index=index_paths, columns=cols, dtype='float64')
    df.to_csv(out_csv)
    # DLC typically stores a key named 'df_with_missing'.
    # Writing HDF5 requires PyTables (python package 'tables').
    try:
        df.to_hdf(out_h5, key='df_with_missing', mode='w')
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[DLC] Warning: could not write HDF5 '{out_h5}' (PyTables missing: {e}). CSV was written.")
        print("[DLC] Install PyTables to also write CollectedData_*.h5: conda install -c conda-forge pytables  (or: pip install tables)")

def main():
    ap = argparse.ArgumentParser(description="Deform textured mesh using Mouse 1 keypoints and render videos (RBF+pyrender).")
    ap.add_argument("--mesh", required=True, help="Path to mesh file (.obj or .ply). For OBJ, keep .mtl and textures alongside.")
    ap.add_argument("--mesh-nodes", required=True, help="mouse_mesh_nodes.txt: keypoint coords in mesh coordinate system ([NODES] section).")
    ap.add_argument("--coords-3d", required=True, help="coords_3d.csv output from mouse-sim.py / mouse_sim2.py.")
    ap.add_argument("--mouse-ids", default="all", help="Comma-separated mouse IDs to render (e.g. '0,1') or 'all'.")
    ap.add_argument("--cameras", required=True, help="cameras.json exported by mouse-sim.py.")
    ap.add_argument("--out-dir", default="out_render", help="Output directory.")
    ap.add_argument("--fps", type=float, default=10.0, help="FPS for output videos.")

    ap.add_argument("--kernel", default="thin_plate_spline",
                    choices=["thin_plate_spline", "cubic", "quintic", "linear", "multiquadric", "inverse_multiquadric", "gaussian"],
                    help="RBF kernel.")
    ap.add_argument("--anchors", type=int, default=8,
                    help="Number of anchor points (0 disables). Currently: 8 means bbox corners.")
    ap.add_argument("--anchor-scale", type=float, default=1.25,
                    help="BBox anchor expansion scale around mesh.")
    ap.add_argument("--rigid-fit-nodes", default="head,nose_tip,left_ear_tip,right_ear_tip,tail_root,tail_tip,left_front_paw,right_front_paw,left_hind_paw,right_hind_paw",
                    help="Comma-separated labels for rigid base fit, or 'all'.")

    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)

    ap.add_argument("--T", default=None, help="Top face image (z=height)")
    ap.add_argument("--B", default=None, help="Bottom face image (z=0)")
    ap.add_argument("--N", default=None, help="North wall image (y=depth)")
    ap.add_argument("--S", default=None, help="South wall image (y=0)")
    ap.add_argument("--E", default=None, help="East wall image (x=width)")
    ap.add_argument("--W", default=None, help="West wall image (x=0)")
    ap.add_argument("--cage-alpha", type=float, default=1.0, help="Face opacity 0..1")

    ap.add_argument("--mouse-on-top", action="store_true",
                help="Render mouse over cage regardless of depth (non-physical).")

    ap.add_argument("--swap-rb-mouse-id", type=int, default=1,
                help="Mouse ID whose mesh colors are rendered with swapped R/B channels (BGR-as-RGB). Use -1 to disable.")


    # Ground-truth segmentation video export (flat colors per mouse ID)
    ap.add_argument("--seg-video", action="store_true",
                help="Also write per-camera ground-truth segmentation videos. Background is black; mice are encoded as flat colors.")
    ap.add_argument("--seg-suffix", default="_seg",
                help="Filename suffix for segmentation videos (before .mp4).")
    ap.add_argument("--seg-cage-front-tol", type=float, default=1e-4,
                help="Physical mode only: a pixel is labeled as mouse only if it is closer than the cage depth by at least this tolerance (scene units).")
    ap.add_argument("--mouse0-bgr", type=parse_bgr, default=parse_bgr("25,35,40"),
                help='Mouse0 color as B,G,R (default "25,35,40").')
    ap.add_argument("--mouse1-bgr", type=parse_bgr, default=parse_bgr("40,35,25"),
                help='Mouse1 color as B,G,R (default "40,35,25").')
    # Ground-truth segmentation label-ID video export (single-channel IDs)
    ap.add_argument("--seg-id-video", action="store_true",
                help="Also write per-camera ground-truth segmentation label-ID videos (single-channel). Background=0; each mouse gets a stable ID starting at 1. Written losslessly as FFV1 .mkv by default.")
    ap.add_argument("--seg-id-suffix", default="_seg_id",
                help="Filename suffix for label-ID videos (before extension).")
    ap.add_argument("--seg-id-ext", default="mkv",
                help="Container extension for label-ID videos. Recommended: mkv (lossless FFV1).")
    ap.add_argument("--seg-id-fourcc", default="FFV1",
                help="FOURCC for label-ID videos. Recommended: FFV1 for lossless.")
    ap.add_argument("--seg-id-vis-video", action="store_true",
                help="Also write a view-friendly grayscale preview video of the label IDs (values remapped to 0..255). This is NOT suitable as ground truth for training.")
    ap.add_argument("--seg-id-vis-suffix", default="_seg_id_vis",
                help="Filename suffix for the view-friendly label-ID preview video.")



    # DLC (DeepLabCut) multi-animal project export
    ap.add_argument("--dlc-project", action="store_true",
                    help="Generate a DeepLabCut multi-animal project (config.yaml + labeled-data + videos).")
    ap.add_argument("--dlc-task", default="mouse_sim_synth", help="DLC project task name.")
    ap.add_argument("--dlc-scorer", default="synthetic", help="DLC scorer name.")
    ap.add_argument("--dlc-every", type=int, default=1,
                    help="Save/label every Nth frame for DLC (1 = every frame).")
    ap.add_argument("--dlc-depth-tol", type=float, default=None,
                    help="Deprecated alias for --dlc-occ-front-tol (kept for backward compatibility).")
    ap.add_argument("--dlc-occ-front-tol", type=float, default=5.0,
                    help="Front-occluder tolerance (scene units). A point is occluded only if rendered depth is closer than z_point - tol.")
    ap.add_argument("--dlc-back-tol", type=float, default=-1.0,
                    help="Optional back-depth tolerance (scene units). <0 disables upper depth check (recommended for thin/subpixel structures).")
    ap.add_argument("--dlc-vis-patch", type=int, default=1,
                    help="Visibility patch radius in pixels (0=single pixel, 1=3x3, 2=5x5).")
    ap.add_argument("--dlc-overwrite", action="store_true",
                    help="Overwrite existing DLC config/labels/frames if present.")

    ap.add_argument("--dlc-check-video", action="store_true",
                    help="Write per-camera overlay videos showing visible bodyparts as filled blobs and occluded bodyparts as hollow circles.")
    ap.add_argument("--dlc-check-suffix", default="_dlc_check",
                    help="Filename suffix for the DLC overlay videos (before .mp4).")

    args = ap.parse_args()
    seg_enabled = bool(getattr(args, 'seg_video', False))
    seg_id_enabled = bool(getattr(args, "seg_id_video", False))
    seg_id_vis_enabled = bool(getattr(args, "seg_id_vis_video", False))
    seg_any_enabled = seg_enabled or seg_id_enabled or seg_id_vis_enabled
    os.makedirs(args.out_dir, exist_ok=True)

    dlc_enabled = bool(args.dlc_project)
    dlc_project_dir = os.path.abspath(args.out_dir) if dlc_enabled else ""
    dlc_videos_dir = os.path.join(dlc_project_dir, "videos") if dlc_enabled else ""
    dlc_labeled_dir = os.path.join(dlc_project_dir, "labeled-data") if dlc_enabled else ""
    if dlc_enabled:
        os.makedirs(dlc_videos_dir, exist_ok=True)
        os.makedirs(dlc_labeled_dir, exist_ok=True)
    if args.dlc_every < 1:
        raise ValueError("--dlc-every must be >= 1")
    if args.dlc_depth_tol is not None:
        # Backward-compatible alias.
        args.dlc_occ_front_tol = float(args.dlc_depth_tol)
    if args.dlc_vis_patch < 0:
        raise ValueError("--dlc-vis-patch must be >= 0")
    dlc_video_paths: Dict[str, str] = {}

    meshes = load_meshes(args.mesh)
    swapped_visuals = None
    if args.swap_rb_mouse_id is not None and args.swap_rb_mouse_id >= 0:
        swapped_visuals = [make_swapped_visual(m.visual) for m in meshes]

    mesh_nodes = {norm_label(k): v for k, v in load_nodes_txt(args.mesh_nodes).items()}
    frames = load_coords_3d_csv(args.coords_3d)
    cage, cams = load_scene_json(args.cameras)

    # Mouse selection (for multi-mouse coords_3d.csv). Backward compatible with single-mouse.
    all_mouse_ids = sorted({mid for fr in frames.values() for mid in fr.mice.keys()})
    if not all_mouse_ids:
        raise ValueError("No mouse data found in coords_3d.csv")
    if args.mouse_ids.strip().lower() == "all":
        selected_mouse_ids: Optional[Set[int]] = None
    else:
        selected_mouse_ids = {int(s) for s in args.mouse_ids.split(",") if s.strip() != ""}
        if not selected_mouse_ids:
            selected_mouse_ids = None
    print("mice in data:", all_mouse_ids, "selected:", ("all" if selected_mouse_ids is None else sorted(selected_mouse_ids)))

    dlc_mouse_ids = all_mouse_ids if selected_mouse_ids is None else sorted(selected_mouse_ids)

    # Stable label IDs for segmentation ID export: background=0, mice start at 1 (sorted by mouse id).

    seg_label_by_mid = {mid: (i + 1) for i, mid in enumerate(sorted(dlc_mouse_ids))}

    seg_max_label = max(seg_label_by_mid.values()) if seg_label_by_mid else 0

    # View-friendly mapping (0..seg_max_label) -> (0..255). Used only for visualization video.

    seg_vis_lut = (np.linspace(0, 255, seg_max_label + 1).round().astype(np.uint8) if seg_max_label > 0 else np.array([0], dtype=np.uint8))
    dlc_individuals = [f"mouse{mid}" for mid in dlc_mouse_ids]

    # Determine frame range
    all_frame_ids = sorted(frames.keys())
    f0 = all_frame_ids[0] if args.start_frame is None else args.start_frame
    f1 = all_frame_ids[-1] if args.end_frame is None else args.end_frame
    frame_ids = [fi for fi in all_frame_ids if f0 <= fi <= f1]
    if not frame_ids:
        raise ValueError("No frames selected.")

    # Constraint labels: all nodes present in mesh_nodes (exact constraints)
    constraint_labels = sorted(mesh_nodes.keys())

    # Rigid fit labels
    if args.rigid_fit_nodes.strip().lower() == "all":
        fit_labels = constraint_labels
    else:
        fit_labels = [norm_label(s) for s in args.rigid_fit_nodes.split(",") if s.strip()]
    # validate
    for lab in constraint_labels:
        if lab not in mesh_nodes:
            raise ValueError(f"Missing in mesh_nodes: {lab}")
    for lab in fit_labels:
        if lab not in mesh_nodes:
            raise ValueError(f"Rigid-fit label not found in mesh_nodes: {lab}")

    # DLC bodyparts = intersection(coords_3d nodes, mesh_nodes)
    coords_nodes: Set[str] = set()
    for _fi in frame_ids:
        _fr = frames[_fi]
        for _mid, _mfr in _fr.mice.items():
            if selected_mouse_ids is not None and _mid not in selected_mouse_ids:
                continue
            coords_nodes.update(_mfr.pts.keys())
    dlc_bodyparts = sorted(coords_nodes.intersection(set(mesh_nodes.keys())))
    if dlc_enabled:
        only_coords = sorted(coords_nodes - set(mesh_nodes.keys()))
        only_mesh = sorted(set(mesh_nodes.keys()) - coords_nodes)
        if only_coords or only_mesh:
            print("[DLC] Warning: coords_3d and mesh_nodes differ; using intersection.")
            if only_coords:
                print("[DLC] nodes only in coords_3d (ignored):", only_coords[:20], ("..." if len(only_coords) > 20 else ""))
            if only_mesh:
                print("[DLC] nodes only in mesh_nodes (ignored):", only_mesh[:20], ("..." if len(only_mesh) > 20 else ""))
        if not dlc_bodyparts:
            raise ValueError("[DLC] No bodyparts in intersection(coords_3d, mesh_nodes)")

    # Anchors (mesh space)
    anchors_mesh = None
    if args.anchors != 0:
        # currently only supports 8 bbox anchors
        anchors_mesh = mesh_bbox_anchors(meshes, scale=args.anchor_scale)

    # Setup renderers + video writers
    scenes_cage = {}
    scenes_mouse = {}
    renderers = {}
    writers = {}
    dlc_check_writers: Dict[str, cv2.VideoWriter] = {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    pre_cage_rgb = {}
    pre_cage_depth: Dict[str, np.ndarray] = {}
    seg_writers: Dict[str, cv2.VideoWriter] = {}

    seg_id_writers: Dict[str, cv2.VideoWriter] = {}
    seg_id_vis_writers: Dict[str, cv2.VideoWriter] = {}
    for cam in cams:
        # One renderer per camera
        renderer = make_renderer_for_camera(cam)

        # Two scenes sharing the same camera+lights configuration
        scene_cage = make_scene_for_camera(cam, bg_color=(30, 30, 30), ambient=(0.8, 0.8, 0.8))
        scene_mouse = make_scene_for_camera(cam, bg_color=(0, 0, 0), ambient=(0.8, 0.8, 0.8))

        if cage is not None:
            tex = {"T": args.T, "B": args.B, "N": args.N, "S": args.S, "E": args.E, "W": args.W}
            Cw = camera_center_world(cam.R_wc, cam.t_wc)
            fw = camera_forward_world_cv(cam.R_wc)
            skip_face = pick_face_to_skip(Cw, fw, cage)
            add_cage_to_scene(scene_cage, cage, tex, alpha=args.cage_alpha, skip_face=skip_face)

        scenes_cage[cam.name] = scene_cage
        scenes_mouse[cam.name] = scene_mouse
        renderers[cam.name] = renderer

        # ✅ pre-render cage for EVERY camera (when mouse-on-top mode is enabled)
        if args.mouse_on_top and cage is not None:
            rgb, _ = renderer.render(scene_cage)
            pre_cage_rgb[cam.name] = rgb

        # Pre-render cage depth once per camera for segmentation in physical mode.
        if seg_any_enabled and (not args.mouse_on_top) and (cage is not None):
            _, d_cage = renderer.render(scene_cage)
            pre_cage_depth[cam.name] = d_cage

        out_path = os.path.join(dlc_videos_dir if dlc_enabled else args.out_dir, f"{cam.name}.mp4")
        if dlc_enabled:
            dlc_video_paths[cam.name] = out_path
        writers[cam.name] = cv2.VideoWriter(out_path, fourcc, args.fps, (cam.width, cam.height))

        if seg_enabled:
            seg_path = os.path.join(dlc_videos_dir if dlc_enabled else args.out_dir, f"{cam.name}{args.seg_suffix}.mp4")
            seg_writers[cam.name] = cv2.VideoWriter(seg_path, fourcc, args.fps, (cam.width, cam.height))


        if seg_id_enabled:

            seg_id_ext = args.seg_id_ext.lstrip(".")

            seg_id_path = os.path.join(dlc_videos_dir if dlc_enabled else args.out_dir, f"{cam.name}{args.seg_id_suffix}.{seg_id_ext}")

            seg_id_fourcc = cv2.VideoWriter_fourcc(*str(args.seg_id_fourcc)[:4])

            seg_id_writers[cam.name] = cv2.VideoWriter(seg_id_path, seg_id_fourcc, args.fps, (cam.width, cam.height), isColor=False)


        if seg_id_vis_enabled:

            seg_id_vis_path = os.path.join(dlc_videos_dir if dlc_enabled else args.out_dir, f"{cam.name}{args.seg_id_vis_suffix}.mp4")

            seg_id_vis_writers[cam.name] = cv2.VideoWriter(seg_id_vis_path, fourcc, args.fps, (cam.width, cam.height), isColor=False)
        if dlc_enabled and args.dlc_check_video:
            check_base = f"{cam.name}{args.dlc_check_suffix}.mp4"
            check_path = os.path.join(dlc_videos_dir, check_base)
            dlc_check_writers[cam.name] = cv2.VideoWriter(check_path, fourcc, args.fps, (cam.width, cam.height))

    # DLC project metadata + per-camera tables
    dlc_index_paths_by_cam: Dict[str, List[str]] = {}
    dlc_rows_by_cam: Dict[str, List[List[float]]] = {}
    dlc_cam_frame_dirs: Dict[str, str] = {}
    if dlc_enabled:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        for cam in cams:
            cam_ld = os.path.join(dlc_labeled_dir, cam.name)
            os.makedirs(cam_ld, exist_ok=True)
            dlc_cam_frame_dirs[cam.name] = cam_ld
            dlc_index_paths_by_cam[cam.name] = []
            dlc_rows_by_cam[cam.name] = []
        dlc_cfg = _make_dlc_config(
            project_dir=dlc_project_dir,
            task=args.dlc_task,
            scorer=args.dlc_scorer,
            date_str=date_str,
            individuals=dlc_individuals,
            bodyparts=dlc_bodyparts,
            cams=cams,
            video_paths=dlc_video_paths,
        )
        _write_dlc_project(dlc_project_dir, dlc_cfg, overwrite=args.dlc_overwrite)

    # Pre-pack mesh keypoints arrays (mesh space)
    Xk_mesh = np.stack([mesh_nodes[l] for l in constraint_labels], axis=0)
    Xfit_mesh = np.stack([mesh_nodes[l] for l in fit_labels], axis=0)

    print("pre_cage_rgb keys:", sorted(pre_cage_rgb.keys()))
    print("camera names:", [c.name for c in cams])

    # Render loop
    for idx, fi in enumerate(frame_ids):
        fr = frames[fi]
        frame_bgr_by_cam: Dict[str, np.ndarray] = {}
        depth_for_dlc_by_cam: Dict[str, np.ndarray] = {}

        # Determine which mice to render for this frame
        mouse_ids = sorted(fr.mice.keys())
        if selected_mouse_ids is not None:
            mouse_ids = [mid for mid in mouse_ids if mid in selected_mouse_ids]
        if not mouse_ids:
            raise ValueError(f"Frame {fi} has no mice matching --mouse-ids")

        overlay_bgr_by_cam: Dict[str, np.ndarray] = {}

        # Deform meshes per mouse
        deformed_by_mouse: Dict[int, List[trimesh.Trimesh]] = {}

        for mid in mouse_ids:
            mfr = fr.mice[mid]

            missing = [l for l in constraint_labels if l not in mfr.pts]
            if missing:
                raise ValueError(f"Frame {fi} mouse {mid} missing keypoints: {missing}")

            Yk_world = np.stack([mfr.pts[l] for l in constraint_labels], axis=0)
            Yfit_world = np.stack([mfr.pts[l] for l in fit_labels], axis=0)

            deformed_trimeshes: List[trimesh.Trimesh] = []
            use_swapped = (swapped_visuals is not None and mid == args.swap_rb_mouse_id)
            for mi, m in enumerate(meshes):
                V_def = deform_vertices_rigid_rbf(
                    V_mesh=np.asarray(m.vertices, dtype=np.float64),
                    Xk_mesh=Xk_mesh,
                    Yk_world=Yk_world,
                    Xfit_mesh=Xfit_mesh,
                    Yfit_world=Yfit_world,
                    anchors_mesh=anchors_mesh,
                    kernel=args.kernel,
                )
                visual = swapped_visuals[mi] if use_swapped else m.visual
                dm = trimesh.Trimesh(
                    vertices=V_def,
                    faces=m.faces,
                    visual=visual,
                    process=False,
                    maintain_order=True,
                )
                deformed_trimeshes.append(dm)

            deformed_by_mouse[mid] = deformed_trimeshes

        # For each camera: add mesh nodes, render, remove, write frame
        for cam in cams:
            renderer = renderers[cam.name]

            if args.mouse_on_top and cage is not None:
                # --- Pass 1: render cage only (static)
                cage_rgb = pre_cage_rgb.get(cam.name)
                if cage_rgb is None:
                    # fallback (also “self-heals” if you forgot to precompute)
                    cage_rgb, _ = renderer.render(scenes_cage[cam.name])
                    pre_cage_rgb[cam.name] = cage_rgb


                # --- Pass 2: render mouse only (no cage in this scene)
                scene_mouse = scenes_mouse[cam.name]
                node_handles = []
                for mid in mouse_ids:
                    for dm in deformed_by_mouse[mid]:
                        pr_mesh = trimesh_to_pyrender(dm)
                        nh = scene_mouse.add(pr_mesh, pose=np.eye(4))
                        node_handles.append(nh)

                mouse_rgb, mouse_depth = renderer.render(scene_mouse)

                # remove mouse nodes
                for nh in node_handles:
                    scene_mouse.remove_node(nh)

                # composite: wherever mouse rendered, overwrite cage pixels
                mask = mouse_depth > 0
                out_rgb = cage_rgb.copy()
                out_rgb[mask] = mouse_rgb[mask]

            else:
                # --- Physical mode: render mouse inside the cage scene (normal depth test)
                scene = scenes_cage[cam.name]  # this scene already contains the cage
                node_handles = []
                for mid in mouse_ids:
                    for dm in deformed_by_mouse[mid]:
                        pr_mesh = trimesh_to_pyrender(dm)
                        nh = scene.add(pr_mesh, pose=np.eye(4))
                        node_handles.append(nh)

                out_rgb, scene_depth = renderer.render(scene)

                # remove mouse nodes
                for nh in node_handles:
                    scene.remove_node(nh)

            # pyrender gives RGB; cv2 wants BGR
            bgr = out_rgb[..., ::-1].copy()

            writers[cam.name].write(bgr)

            # Ground-truth segmentation (flat colors per mouse ID)
            if seg_any_enabled and ((cam.name in seg_writers) or (cam.name in seg_id_writers) or (cam.name in seg_id_vis_writers)):
                H, W = cam.height, cam.width
                depth_best = np.full((H, W), np.inf, dtype=np.float32)
                best_mid = np.full((H, W), -1, dtype=np.int32)

                # Choose base scene for depth testing.
                # - mouse_on_top: ignore cage depth entirely (matches the composited RGB output)
                # - physical mode: include cage for occlusion, but label only mouse pixels
                base_scene = scenes_mouse[cam.name] if (args.mouse_on_top or cage is None) else scenes_cage[cam.name]

                # In physical mode with a cage, we need the cage-only depth to decide whether a pixel is cage vs mouse.
                cage_depth = None
                if (not args.mouse_on_top) and (cage is not None):
                    cage_depth = pre_cage_depth.get(cam.name)
                    if cage_depth is None:
                        # fallback / self-heal if not precomputed
                        _, cage_depth = renderer.render(scenes_cage[cam.name])
                        pre_cage_depth[cam.name] = cage_depth

                for mid in mouse_ids:
                    node_handles_seg = []
                    for dm in deformed_by_mouse[mid]:
                        pr_mesh = trimesh_to_pyrender(dm)
                        nh = base_scene.add(pr_mesh, pose=np.eye(4))
                        node_handles_seg.append(nh)

                    _, d_full = renderer.render(base_scene)

                    for nh in node_handles_seg:
                        base_scene.remove_node(nh)

                    if args.mouse_on_top or cage is None or cage_depth is None:
                        mask_mid = (d_full > 0)
                    else:
                        # A pixel belongs to the mouse only if it is in front of the cage at that pixel.
                        # This suppresses mouse pixels occluded by cage walls/floor/ceiling.
                        mask_mid = (d_full > 0) & ((cage_depth == 0) | (d_full < (cage_depth - args.seg_cage_front_tol)))

                    closer = mask_mid & (d_full < depth_best)
                    depth_best[closer] = d_full[closer]
                    best_mid[closer] = mid

                # Build label-ID frame once if any ID-based output is enabled for this camera.
                need_seg_id = ((seg_id_enabled and (cam.name in seg_id_writers)) or (seg_id_vis_enabled and (cam.name in seg_id_vis_writers)))
                seg_id = None
                if need_seg_id:
                    seg_id = np.zeros((H, W), dtype=np.uint8)
                    for mid in set(mouse_ids):
                        m = (best_mid == mid)
                        if np.any(m):
                            seg_id[m] = np.uint8(seg_label_by_mid.get(mid, 0))

                # Color segmentation (flat colors) for visualization
                if cam.name in seg_writers:
                    seg_bgr = np.zeros((H, W, 3), dtype=np.uint8)  # background = black
                    for mid in set(mouse_ids):
                        m = (best_mid == mid)
                        if np.any(m):
                            seg_bgr[m] = mouse_id_to_bgr(mid, args.mouse0_bgr, args.mouse1_bgr)
                    seg_writers[cam.name].write(seg_bgr)

                # Label-ID segmentation (single-channel): exact IDs, lossless codec recommended
                if (seg_id is not None) and (cam.name in seg_id_writers):
                    seg_id_writers[cam.name].write(seg_id)

                # View-friendly grayscale preview (values remapped to 0..255; NOT for training)
                if (seg_id is not None) and (cam.name in seg_id_vis_writers):
                    seg_vis = seg_vis_lut[seg_id]
                    seg_id_vis_writers[cam.name].write(seg_vis)
            if dlc_enabled:
                frame_bgr_by_cam[cam.name] = bgr
                if args.mouse_on_top:
                    depth_for_dlc_by_cam[cam.name] = mouse_depth
                else:
                    # Depth includes the full scene; best-effort fallback when mouse_on_top is disabled.
                    depth_for_dlc_by_cam[cam.name] = scene_depth

                if args.dlc_check_video:
                    overlay_bgr = _draw_dlc_overlay_markers(
                        bgr=bgr,
                        cam=cam,
                        fr=fr,
                        mouse_ids=dlc_mouse_ids,
                        bodyparts=dlc_bodyparts,
                        depth_map=depth_for_dlc_by_cam[cam.name],
                        front_tol=args.dlc_occ_front_tol,
                        vis_patch_radius=args.dlc_vis_patch,
                        back_tol=(None if args.dlc_back_tol < 0 else args.dlc_back_tol),
                    )
                    overlay_bgr_by_cam[cam.name] = overlay_bgr
                    if cam.name in dlc_check_writers:
                        dlc_check_writers[cam.name].write(overlay_bgr)


        # DLC labels/images (sampled frames only)
        if dlc_enabled and (idx % args.dlc_every == 0):
            for cam in cams:
                if cam.name not in frame_bgr_by_cam or cam.name not in depth_for_dlc_by_cam:
                    continue
                img_dir = dlc_cam_frame_dirs[cam.name]
                img_name = f"img{fi:06d}.png"
                img_path = os.path.join(img_dir, img_name)
                if args.dlc_overwrite or (not os.path.exists(img_path)):
                    cv2.imwrite(img_path, frame_bgr_by_cam[cam.name])

                depth_map = depth_for_dlc_by_cam[cam.name]
                row_xy: List[float] = []
                for mid in dlc_mouse_ids:
                    mfr = fr.mice.get(mid, None)
                    for bp in dlc_bodyparts:
                        if mfr is None or bp not in mfr.pts:
                            row_xy.extend([np.nan, np.nan])
                            continue
                        Pw = mfr.pts[bp]
                        u, v, z = _project_point_cv(cam, Pw)
                        vis = _is_visible_from_depth(u, v, z, depth_map, front_tol=args.dlc_occ_front_tol, patch_radius=args.dlc_vis_patch, back_tol=(None if args.dlc_back_tol < 0 else args.dlc_back_tol))
                        if vis:
                            row_xy.extend([float(u), float(v)])
                        else:
                            row_xy.extend([np.nan, np.nan])

                rel_idx = _posix_relpath(img_path, dlc_project_dir)
                dlc_index_paths_by_cam[cam.name].append(rel_idx)
                dlc_rows_by_cam[cam.name].append(row_xy)


        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_ids):
            print(f"Rendered {idx+1}/{len(frame_ids)} frames")

    # cleanup
    for cam in cams:
        writers[cam.name].release()
        if cam.name in seg_writers:
            seg_writers[cam.name].release()

        if cam.name in seg_id_writers:
            seg_id_writers[cam.name].release()
        if cam.name in seg_id_vis_writers:
            seg_id_vis_writers[cam.name].release()
        if cam.name in dlc_check_writers:
            dlc_check_writers[cam.name].release()
        renderers[cam.name].delete()

    if dlc_enabled:
        for cam in cams:
            cam_dir = dlc_cam_frame_dirs[cam.name]
            out_csv = os.path.join(cam_dir, f"CollectedData_{args.dlc_scorer}.csv")
            out_h5 = os.path.join(cam_dir, f"CollectedData_{args.dlc_scorer}.h5")
            _write_dlc_collected_data(
                out_csv=out_csv,
                out_h5=out_h5,
                scorer=args.dlc_scorer,
                individuals=dlc_individuals,
                bodyparts=dlc_bodyparts,
                index_paths=dlc_index_paths_by_cam[cam.name],
                rows_xy=dlc_rows_by_cam[cam.name],
            )
        print(f"[DLC] Project written to: {dlc_project_dir}")
        print(f"[DLC] individuals={dlc_individuals} bodyparts={len(dlc_bodyparts)} labeled_every={args.dlc_every}")
        if args.dlc_check_video:
            print(f"[DLC] overlay videos written with suffix '{args.dlc_check_suffix}.mp4' in: {dlc_videos_dir}")

    if seg_enabled:
        print(f"[SEG] segmentation videos written with suffix '{args.seg_suffix}.mp4' in: {dlc_videos_dir if dlc_enabled else args.out_dir}")



    if seg_id_enabled:
        out_dir = dlc_videos_dir if dlc_enabled else args.out_dir
        print(f"[SEG-ID] label-ID videos written with suffix '{args.seg_id_suffix}.{args.seg_id_ext.lstrip('.')}' in: {out_dir}")
    if seg_id_vis_enabled:
        out_dir = dlc_videos_dir if dlc_enabled else args.out_dir
        print(f"[SEG-ID] preview videos written with suffix '{args.seg_id_vis_suffix}.mp4' in: {out_dir}")
    print("Done. Videos in:", args.out_dir)


if __name__ == "__main__":
    main()
