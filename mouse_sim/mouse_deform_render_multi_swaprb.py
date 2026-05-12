#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy.interpolate import RBFInterpolator

from PIL import Image

import trimesh
import pyrender

import copy

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("OpenCV is required for writing MP4. Install opencv from conda-forge.") from e


def norm_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


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

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    pre_cage_rgb = {}

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

        out_path = os.path.join(args.out_dir, f"{cam.name}.mp4")
        writers[cam.name] = cv2.VideoWriter(out_path, fourcc, args.fps, (cam.width, cam.height))

    # Pre-pack mesh keypoints arrays (mesh space)
    Xk_mesh = np.stack([mesh_nodes[l] for l in constraint_labels], axis=0)
    Xfit_mesh = np.stack([mesh_nodes[l] for l in fit_labels], axis=0)

    print("pre_cage_rgb keys:", sorted(pre_cage_rgb.keys()))
    print("camera names:", [c.name for c in cams])

    # Render loop
    for idx, fi in enumerate(frame_ids):
        fr = frames[fi]

        # Determine which mice to render for this frame
        mouse_ids = sorted(fr.mice.keys())
        if selected_mouse_ids is not None:
            mouse_ids = [mid for mid in mouse_ids if mid in selected_mouse_ids]
        if not mouse_ids:
            raise ValueError(f"Frame {fi} has no mice matching --mouse-ids")

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

                out_rgb, _ = renderer.render(scene)

                # remove mouse nodes
                for nh in node_handles:
                    scene.remove_node(nh)

            # pyrender gives RGB; cv2 wants BGR
            bgr = out_rgb[..., ::-1].copy()
            writers[cam.name].write(bgr)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_ids):
            print(f"Rendered {idx+1}/{len(frame_ids)} frames")

    # cleanup
    for cam in cams:
        writers[cam.name].release()
        renderers[cam.name].delete()

    print("Done. Videos in:", args.out_dir)


if __name__ == "__main__":
    main()

