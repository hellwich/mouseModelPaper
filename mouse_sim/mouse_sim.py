#!/usr/bin/env python3
"""
Mouse 1.0 — Option A (simple trot + tightened rearing + behavior protocol)

Changes vs previous version:
- Adds behavior_protocol.csv: records every switch 0 -> 1 (running -> rearing)
- Tightens rearing: rotates the body about tail_root so the tail_root->head axis is
  approximately vertical at peak of rearing (u=0.5), then returns.

Compromises remain (as agreed for Option A):
- During running, paws move on ground plane z=0 by local-x offsets (trot timing).
  This does not preserve head–paw / tailroot–hindpaw edge lengths.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional video backend
_VIDEO_BACKEND = None
try:
    import cv2  # type: ignore
    _VIDEO_BACKEND = "opencv"
except Exception:
    cv2 = None  # type: ignore

if _VIDEO_BACKEND is None:
    try:
        import imageio  # type: ignore
        _VIDEO_BACKEND = "imageio"
    except Exception:
        imageio = None  # type: ignore


# ----------------------------
# File parsing (mouse graph)
# ----------------------------

def _norm_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


@dataclass
class MouseGraph:
    nodes: Dict[str, np.ndarray]          # label -> (3,)
    edges: List[Tuple[str, str]]          # (a,b)
    node_order: List[str]

    def require(self, label: str) -> str:
        key = _norm_label(label)
        if key not in self.nodes:
            raise ValueError(f"Mouse graph missing required node: {label}")
        return key


def load_mouse_graph(path: str) -> MouseGraph:
    """
    Minimal, forgiving format:

    [NODES]
    head  0.0 0.0 2.4
    nose_tip  1.2 0.0 2.2
    ...
    [EDGES]
    head nose_tip
    head left_ear_tip
    ...

    Lines starting with # are comments. Empty lines ignored.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mode = None
    nodes: Dict[str, np.ndarray] = {}
    edges: List[Tuple[str, str]] = []
    node_order: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper() == "[NODES]":
                mode = "nodes"
                continue
            if line.upper() == "[EDGES]":
                mode = "edges"
                continue
            if mode == "nodes":
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Bad node line: {line}")
                label = _norm_label(parts[0])
                x, y, z = map(float, parts[1:])
                if label in nodes:
                    raise ValueError(f"Duplicate node label: {label}")
                nodes[label] = np.array([x, y, z], dtype=np.float64)
                node_order.append(label)
            elif mode == "edges":
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad edge line: {line}")
                a, b = _norm_label(parts[0]), _norm_label(parts[1])
                edges.append((a, b))
            else:
                raise ValueError(f"File must start with [NODES] section. Got: {line}")

    if not nodes:
        raise ValueError("No nodes loaded.")
    if not edges:
        raise ValueError("No edges loaded.")

    return MouseGraph(nodes=nodes, edges=edges, node_order=node_order)


# ----------------------------
# Math helpers
# ----------------------------

def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def roty(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)

def rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]], dtype=np.float64)

def rotate_leg_vec_about_y_to_reach_z(v0: np.ndarray, z_rel_target: float) -> np.ndarray:
    """
    Rotate vector v0 around the local y-axis (using the same x-z convention as roty)
    so that the rotated vector's z component equals z_rel_target (as closely as possible),
    while preserving vector length. y component is unchanged by y-rotation.

    Returns rotated vector v'.
    """
    x = float(v0[0]); y = float(v0[1]); z = float(v0[2])
    R = math.hypot(x, z)
    if R < 1e-12:
        return v0.copy()

    # Clamp to reachable range in x-z plane
    zt = max(-R, min(R, z_rel_target))

    # We have z' = -sin(b)*x + cos(b)*z = R * cos(b + beta0)
    # where beta0 satisfies cos(beta0)=z/R, sin(beta0)=x/R  => beta0 = atan2(x, z)
    beta0 = math.atan2(x, z)
    ang = math.acos(zt / R)

    b1 = -beta0 + ang
    b2 = -beta0 - ang

    def xprime(b: float) -> float:
        return math.cos(b)*x + math.sin(b)*z

    # Choose the solution that retracts more (smaller x')
    b = b1 if xprime(b1) < xprime(b2) else b2

    xp = math.cos(b)*x + math.sin(b)*z
    zp = -math.sin(b)*x + math.cos(b)*z
    return np.array([xp, y, zp], dtype=np.float64)

# ----------------------------
# Random track model (smooth heading + speed)
# ----------------------------

@dataclass
class TrackState:
    x: float
    y: float
    theta: float   # yaw heading
    v: float       # speed
    omega: float   # yaw rate


def step_track(
    st: TrackState,
    dt: float,
    cage_w: float,
    cage_d: float,
    max_speed: float,
    omega_max: float,
    speed_noise: float,
    omega_noise: float,
    speed_damp: float,
    omega_damp: float,
    wall_repulse_gain: float,
    wall_repulse_dist: float,
    rng: random.Random,
) -> TrackState:
    # OU-ish update
    st.v += (-speed_damp * st.v) * dt + speed_noise * math.sqrt(dt) * rng.gauss(0, 1)
    st.omega += (-omega_damp * st.omega) * dt + omega_noise * math.sqrt(dt) * rng.gauss(0, 1)

    st.v = max(0.0, min(max_speed, st.v))
    st.omega = max(-omega_max, min(omega_max, st.omega))

    # wall steering toward center near boundaries
    cx, cy = cage_w * 0.5, cage_d * 0.5
    to_c = math.atan2(cy - st.y, cx - st.x)

    def repulse(d: float) -> float:
        if d <= 1e-6:
            return 1.0
        if d >= wall_repulse_dist:
            return 0.0
        return (wall_repulse_dist - d) / wall_repulse_dist

    dL = st.x
    dR = cage_w - st.x
    dB = st.y
    dT = cage_d - st.y
    prox = max(repulse(dL), repulse(dR), repulse(dB), repulse(dT))

    def ang_diff(a: float, b: float) -> float:
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return d

    steer = -wall_repulse_gain * prox * ang_diff(st.theta, to_c)

    st.theta += (st.omega + steer) * dt

    nx = st.x + st.v * dt * math.cos(st.theta)
    ny = st.y + st.v * dt * math.sin(st.theta)

    # reflect at hard boundaries
    if nx < 0.0 or nx > cage_w:
        st.theta = math.pi - st.theta
        nx = min(max(nx, 0.0), cage_w)
    if ny < 0.0 or ny > cage_d:
        st.theta = -st.theta
        ny = min(max(ny, 0.0), cage_d)

    st.x, st.y = nx, ny
    return st


# ----------------------------
# Behaviors
# ----------------------------

@dataclass
class BehaviorState:
    behavior: int
    t_in_state: float
    state_duration: float
    # running gait
    gait_phase: float  # radians
    # rearing individualized params (sampled at behavior entry)
    rear_pitch_scale: float
    rear_paw_lift: float
    rear_paw_retract: float


def sample_duration(behavior: int, rng: random.Random) -> float:
    if behavior == 0:  # running
        return rng.uniform(2.0, 6.0)
    else:              # rearing
        return rng.uniform(0.6, 1.6)

def next_behavior(current: int, rng: random.Random) -> int:
    if current == 0:
        return 1 if rng.random() < 0.25 else 0
    else:
        return 0


def sample_rearing_params(z_head: float, rng: random.Random) -> Tuple[float, float, float]:
    """
    Size-relative parameters (unit-invariant).
    z_head is the rest-pose head height in local coords (your model ground is z=0).
    """
    pitch_scale = 1.0 + rng.uniform(-0.08, 0.08)  # ±8%

    # Scale paw motion to mouse size:
    paw_lift = 0.30 * z_head * (1.0 + rng.uniform(-0.15, 0.15))
    paw_retract = 0.20 * z_head * (1.0 + rng.uniform(-0.15, 0.15))

    return pitch_scale, paw_lift, paw_retract


def pose_running_option_a(
    g: MouseGraph,
    dt: float,
    v_body: float,
    gait_phase: float,
    stride_len: float,
    head_jitter_amp_deg: float,
    head_jitter_hz: float,
    t_global: float,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Running pose:
    - head-ears-nose cluster stays rigid via small rotations around head
    - paws remain on z=0 and move in local x only, with diagonal trot timing.
    """
    nodes = {k: v.copy() for k, v in g.nodes.items()}

    head = g.require("head")
    nose = g.require("nose_tip")
    le = g.require("left_ear_tip")
    re = g.require("right_ear_tip")
    lfp = g.require("left_front_paw")
    rfp = g.require("right_front_paw")
    lhp = g.require("left_hind_paw")
    rhp = g.require("right_hind_paw")

    # --- rigid head cluster jitter (keeps ears-nose triangle rigid) ---
    A = math.radians(head_jitter_amp_deg)
    w = 2 * math.pi * head_jitter_hz
    pitch = A * math.sin(w * t_global + 0.3)
    roll  = A * 0.7 * math.sin(w * t_global + 1.1)
    yaw   = A * 0.5 * math.sin(w * t_global + 2.0)
    Q = rotz(yaw) @ rotx(roll) @ roty(pitch)

    head_pos = nodes[head].copy()
    for key in (nose, le, re):
        v0 = g.nodes[key] - g.nodes[head]
        nodes[key] = head_pos + Q @ v0

    # --- gait phase advance tied to speed ---
    if stride_len <= 1e-9 or v_body <= 1e-9:
        dphi = 0.0
    else:
        dphi = 2 * math.pi * (v_body / stride_len) * dt
    gait_phase = (gait_phase + dphi) % (2 * math.pi)

    # triangle wave in [-1,1]
    def tri_wave(phase: float) -> float:
        u = (phase % (2 * math.pi)) / (2 * math.pi)
        if u < 0.5:
            return 1.0 - 4.0 * u   # 1 -> -1
        else:
            return -3.0 + 4.0 * u  # -1 -> 1

    # Range in x is about stride_len/2 total (±stride/4)
    def paw_x_offset(phase: float) -> float:
        return (stride_len / 4.0) * tri_wave(phase)

    # diagonal pairing:
    # Pair A: LFP + RHP use phase φ
    # Pair B: RFP + LHP use phase φ+π
    for paw, ph in (
        (lfp, gait_phase),
        (rhp, gait_phase),
        (rfp, gait_phase + math.pi),
        (lhp, gait_phase + math.pi),
    ):
        p0 = g.nodes[paw]
        nodes[paw] = np.array([p0[0] + paw_x_offset(ph), p0[1], 0.0], dtype=np.float64)

    return nodes, gait_phase


def pose_rearing_tight(
    g: MouseGraph,
    t_in_state: float,
    state_duration: float,
    rear_pitch_scale: float,
    rear_paw_lift: float,
    rear_paw_retract: float,
) -> Dict[str, np.ndarray]:
    """
    Tightened rearing (still simple):

    Goal: at peak (u=0.5) rotate the body around tail_root such that
    the vector (tail_root -> head) is ~vertical (parallel to +z),
    then return to rest by u=1.

    Time envelope: h(u) = sin(pi*u), giving 0..1..0.
    """
    nodes = {k: v.copy() for k, v in g.nodes.items()}

    head = g.require("head")
    nose = g.require("nose_tip")
    le = g.require("left_ear_tip")
    re = g.require("right_ear_tip")
    tail_root = g.require("tail_root")
    tail_tip = g.require("tail_tip")
    lfp = g.require("left_front_paw")
    rfp = g.require("right_front_paw")
    lhp = g.require("left_hind_paw")
    rhp = g.require("right_hind_paw")

    # time envelope 0..1..0
    u = 0.0 if state_duration <= 1e-9 else max(0.0, min(1.0, t_in_state / state_duration))
    h = math.sin(math.pi * u)  # 0..1..0

    # Compute pitch_peak so that after rotation the (tail_root->head) vector has x'≈0, z'>0.
    a0 = g.nodes[head] - g.nodes[tail_root]
    ax = float(a0[0])
    az = float(a0[2])

    # With roty() as defined: x' = cos(p)*ax + sin(p)*az
    # Solve x'=0 => p = atan2(-ax, az)
    pitch_peak = math.atan2(-ax, az)

    pitch = (pitch_peak * rear_pitch_scale) * h

    pivot = g.nodes[tail_root]
    Q = roty(pitch)

    # Rotate a body cluster about tail_root; this makes tail_root->head vertical at peak.
    # Include head cluster + tail tip. (tail_root stays fixed).
    body_cluster = (head, nose, le, re, tail_tip)
    for key in body_cluster:
        v0 = g.nodes[key] - pivot
        nodes[key] = pivot + Q @ v0

    # Hind paws stay on ground (cheap): keep them at rest on z=0.
    # (If rest z isn't 0, we force z=0 for this simulation.)
    for key in (lhp, rhp):
        p0 = g.nodes[key]
        nodes[key] = np.array([p0[0], p0[1], 0.0], dtype=np.float64)

    # Front paws: preserve head->paw length by rotating the rest leg vector around head.
    head_pos = nodes[head]  # head after body pitch

    # Desired absolute paw height above ground during rearing (0..rear_paw_lift..0)
    paw_z_abs = rear_paw_lift * h

    for paw in (lfp, rfp):
        v0 = g.nodes[paw] - g.nodes[head]  # rest vector from head to paw (constant length)
        z_rel_target = paw_z_abs - float(head_pos[2])  # desired z relative to head
        v_rot = rotate_leg_vec_about_y_to_reach_z(v0, z_rel_target)
        nodes[paw] = head_pos + v_rot

    return nodes


# ----------------------------
# Cameras and projection
# ----------------------------

@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # world->cam rotation (3x3)
    t: np.ndarray  # world->cam translation (3,)
    pos_world: np.ndarray

    def project(self, Pw: np.ndarray) -> Tuple[float, float, bool]:
        Pc = self.R @ Pw + self.t
        if Pc[2] <= 1e-9:
            return 0.0, 0.0, False
        u = self.fx * (Pc[0] / Pc[2]) + self.cx
        v = self.fy * (Pc[1] / Pc[2]) + self.cy
        visible = (0.0 <= u < self.width) and (0.0 <= v < self.height)
        return float(u), float(v), bool(visible)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    world->camera where cam axes are:
      x: right, y: down, z: forward
    """
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)

    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    true_up = np.cross(right, fwd)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-12)

    Rcw = np.vstack([right, -true_up, fwd])
    R = Rcw
    t = -R @ eye
    return R, t


def make_default_cameras(
    cage_w: float, cage_d: float, cage_h: float,
    image_w: int, image_h: int,
    fov_deg: float
) -> List[Camera]:
    C = np.array([cage_w/2, cage_d/2, cage_h/2], dtype=np.float64)
    corners = []
    for x in (0.0, cage_w):
        for y in (0.0, cage_d):
            for z in (0.0, cage_h):
                corners.append(np.array([x, y, z], dtype=np.float64))
    corners = np.array(corners)

    fov = math.radians(fov_deg)
    fx = (image_w / 2) / math.tan(fov / 2)
    fy = fx
    cx = image_w / 2
    cy = image_h / 2

    r = np.max(np.linalg.norm(corners - C, axis=1))
    dist = r / math.sin(fov / 2) + 1e-6

    cams = []
    # top
    eye1 = np.array([cage_w/2, cage_d/2, cage_h/2 + dist], dtype=np.float64)
    R1, t1 = look_at(eye1, C, np.array([0, 1, 0], dtype=np.float64))
    cams.append(Camera("cam1_top", image_w, image_h, fx, fy, cx, cy, R1, t1, eye1))
    # front (-y)
    eye2 = np.array([cage_w/2, cage_d/2 - dist, cage_h/2 + 0.25*cage_h], dtype=np.float64)
    R2, t2 = look_at(eye2, C, np.array([0, 0, 1], dtype=np.float64))
    cams.append(Camera("cam2_front", image_w, image_h, fx, fy, cx, cy, R2, t2, eye2))
    # side (-x)
    eye3 = np.array([cage_w/2 - dist, cage_d/2, cage_h/2 + 0.25*cage_h], dtype=np.float64)
    R3, t3 = look_at(eye3, C, np.array([0, 0, 1], dtype=np.float64))
    cams.append(Camera("cam3_side", image_w, image_h, fx, fy, cx, cy, R3, t3, eye3))
    return cams

def export_cameras_json(cams, path: str) -> None:
    data = {"cameras": []}
    for cam in cams:
        data["cameras"].append({
            "name": cam.name,
            "width": cam.width,
            "height": cam.height,
            "fx": float(cam.fx),
            "fy": float(cam.fy),
            "cx": float(cam.cx),
            "cy": float(cam.cy),
            # world -> camera: Pc = R*Pw + t
            "R": cam.R.tolist(),
            "t": cam.t.tolist(),
            "pos_world": cam.pos_world.tolist(),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def export_scene_json(cams, path: str, cage_width: float, cage_depth: float, cage_height: float) -> None:
    data = {
        "cage": {
            "origin": [0.0, 0.0, 0.0],
            "width": float(cage_width),
            "depth": float(cage_depth),
            "height": float(cage_height),
        },
        "cameras": []
    }
    for cam in cams:
        data["cameras"].append({
            "name": cam.name,
            "width": cam.width,
            "height": cam.height,
            "fx": float(cam.fx),
            "fy": float(cam.fy),
            "cx": float(cam.cx),
            "cy": float(cam.cy),
            # world -> camera: Pc = R*Pw + t   (CV convention)
            "R": cam.R.tolist(),
            "t": cam.t.tolist(),
            "pos_world": cam.pos_world.tolist(),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# Rendering
# ----------------------------

def render_frame(
    cam: Camera,
    g: MouseGraph,
    world_nodes: Dict[str, np.ndarray],
    cage_w: float,
    cage_d: float,
    cage_h: float,
    draw_labels: bool = False
) -> np.ndarray:
    img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

    if cv2 is None:
        return img

    # cage corners
    cage_pts = []
    for x in (0.0, cage_w):
        for y in (0.0, cage_d):
            for z in (0.0, cage_h):
                cage_pts.append(np.array([x, y, z], dtype=np.float64))
    cage_pts = np.array(cage_pts)

    corner_index = {(x, y, z): i for i, (x, y, z) in enumerate(
        [(x, y, z) for x in (0.0, cage_w) for y in (0.0, cage_d) for z in (0.0, cage_h)]
    )}
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
        ua, va, va_vis = cage_uv[a]
        ub, vb, vb_vis = cage_uv[b]
        if va_vis and vb_vis:
            cv2.line(img, (int(ua), int(va)), (int(ub), int(vb)), (60, 60, 60), 1, cv2.LINE_AA)

    # skeleton edges
    for a, b in g.edges:
        if a not in world_nodes or b not in world_nodes:
            continue
        ua, va, visa = cam.project(world_nodes[a])
        ub, vb, visb = cam.project(world_nodes[b])
        if visa and visb:
            cv2.line(img, (int(ua), int(va)), (int(ub), int(vb)), (220, 220, 220), 2, cv2.LINE_AA)

    # nodes
    for label, Pw in world_nodes.items():
        u, v, vis = cam.project(Pw)
        if not vis:
            continue
        cv2.circle(img, (int(u), int(v)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(img, label, (int(u) + 5, int(v) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return img


# ----------------------------
# Main simulation
# ----------------------------

def simulate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    g = load_mouse_graph(args.mouse_graph)

    # required nodes
    for req in [
        "head", "nose_tip", "left_ear_tip", "right_ear_tip",
        "left_front_paw", "right_front_paw",
        "tail_root", "left_hind_paw", "right_hind_paw", "tail_tip"
    ]:
        g.require(req)

    # stride length = z_head / 3
    z_head = float(g.nodes[_norm_label("head")][2])
    stride_len = z_head / 3.0 if z_head > 0 else args.stride_len_fallback

    os.makedirs(args.out_dir, exist_ok=True)

    # cameras
    cams = make_default_cameras(
        args.cage_width, args.cage_depth, args.cage_height,
        args.image_width, args.image_height,
        args.fov_deg
    )

    if args.export_cameras:
        scene_path = os.path.join(args.out_dir, "cameras.json")
        export_scene_json(cams, scene_path, args.cage_width, args.cage_depth, args.cage_height)

    # writers (video)
    video_writers = {}
    if args.write_video:
        if _VIDEO_BACKEND == "opencv":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            for cam in cams:
                path = os.path.join(args.out_dir, f"{cam.name}.mp4")
                video_writers[cam.name] = cv2.VideoWriter(path, fourcc, args.fps, (cam.width, cam.height))  # type: ignore
        elif _VIDEO_BACKEND == "imageio":
            for cam in cams:
                path = os.path.join(args.out_dir, f"{cam.name}.mp4")
                video_writers[cam.name] = imageio.get_writer(path, fps=args.fps)  # type: ignore
        else:
            print("Video backend not available; will skip video writing.")
            args.write_video = False

    # CSV outputs
    path_3d = os.path.join(args.out_dir, "coords_3d.csv")
    path_protocol = os.path.join(args.out_dir, "behavior_protocol.csv")
    path_2d = {cam.name: os.path.join(args.out_dir, f"keypoints_{cam.name}.csv") for cam in cams}

    f3d = open(path_3d, "w", newline="", encoding="utf-8")
    w3d = csv.writer(f3d)
    w3d.writerow(["frame", "time", "behavior", "node", "x", "y", "z"])

    fproto = open(path_protocol, "w", newline="", encoding="utf-8")
    wproto = csv.writer(fproto)
    # record only 0 -> 1 starts, per your request
    wproto.writerow(["frame", "time", "from_behavior", "to_behavior", "rearing_duration",
                     "rear_pitch_scale", "rear_paw_lift", "rear_paw_retract"])

    f2d = {}
    w2d = {}
    for cam in cams:
        f2d[cam.name] = open(path_2d[cam.name], "w", newline="", encoding="utf-8")
        w2d[cam.name] = csv.writer(f2d[cam.name])
        w2d[cam.name].writerow(["frame", "time", "behavior", "node", "u", "v", "visible"])

    # init track state
    st = TrackState(
        x=args.cage_width * 0.5,
        y=args.cage_depth * 0.5,
        theta=rng.uniform(-math.pi, math.pi),
        v=rng.uniform(0.0, args.max_speed),
        omega=0.0
    )

    # init behavior state
    pitch_scale, paw_lift, paw_retract = sample_rearing_params(z_head, rng)
    beh = BehaviorState(
        behavior=0,
        t_in_state=0.0,
        state_duration=sample_duration(0, rng),
        gait_phase=rng.uniform(0.0, 2 * math.pi),
        rear_pitch_scale=pitch_scale,
        rear_paw_lift=paw_lift,
        rear_paw_retract=paw_retract,
    )

    dt = 1.0 / args.fps
    n_frames = int(round(args.duration * args.fps))

    for frame in range(n_frames):
        t = frame * dt

        # step behavior timer and maybe switch
        beh.t_in_state += dt
        if beh.t_in_state >= beh.state_duration:
            prev = beh.behavior
            newb = next_behavior(beh.behavior, rng)
            beh.behavior = newb
            beh.t_in_state = 0.0
            beh.state_duration = sample_duration(beh.behavior, rng)

            # if entering rearing, sample parameters and write protocol entry (only 0->1)
            if prev == 0 and newb == 1:
                beh.rear_pitch_scale, beh.rear_paw_lift, beh.rear_paw_retract = sample_rearing_params(z_head, rng)
                wproto.writerow([
                    frame, f"{t:.6f}", prev, newb, f"{beh.state_duration:.6f}",
                    f"{beh.rear_pitch_scale:.6f}", f"{beh.rear_paw_lift:.6f}", f"{beh.rear_paw_retract:.6f}"
                ])

        # step track (modulate by behavior)
        max_speed = args.max_speed * (args.rear_speed_scale if beh.behavior == 1 else 1.0)
        omega_noise = args.omega_noise * (args.rear_turn_noise_scale if beh.behavior == 1 else 1.0)

        st = step_track(
            st=st, dt=dt,
            cage_w=args.cage_width, cage_d=args.cage_depth,
            max_speed=max_speed,
            omega_max=args.omega_max,
            speed_noise=args.speed_noise,
            omega_noise=omega_noise,
            speed_damp=args.speed_damp,
            omega_damp=args.omega_damp,
            wall_repulse_gain=args.wall_repulse_gain,
            wall_repulse_dist=args.wall_repulse_dist,
            rng=rng
        )

        # local pose
        if beh.behavior == 0:
            local_nodes, beh.gait_phase = pose_running_option_a(
                g=g,
                dt=dt,
                v_body=st.v,
                gait_phase=beh.gait_phase,
                stride_len=stride_len,
                head_jitter_amp_deg=args.head_jitter_amp_deg,
                head_jitter_hz=args.head_jitter_hz,
                t_global=t,
            )
        else:
            local_nodes = pose_rearing_tight(
                g=g,
                t_in_state=beh.t_in_state,
                state_duration=beh.state_duration,
                rear_pitch_scale=beh.rear_pitch_scale,
                rear_paw_lift=beh.rear_paw_lift,
                rear_paw_retract=beh.rear_paw_retract,
            )

        # transform to world
        R = rotz(st.theta)
        P = np.array([st.x, st.y, 0.0], dtype=np.float64)
        world_nodes = {k: (P + R @ v) for k, v in local_nodes.items()}

        # write 3D
        for node in g.node_order:
            Pw = world_nodes[node]
            w3d.writerow([frame, f"{t:.6f}", beh.behavior, node,
                          f"{Pw[0]:.6f}", f"{Pw[1]:.6f}", f"{Pw[2]:.6f}"])

        # per-camera 2D + video
        for cam in cams:
            for node in g.node_order:
                u, v, vis = cam.project(world_nodes[node])
                w2d[cam.name].writerow([frame, f"{t:.6f}", beh.behavior, node,
                                        f"{u:.6f}", f"{v:.6f}", int(vis)])

            if args.write_video:
                img = render_frame(cam, g, world_nodes,
                                   args.cage_width, args.cage_depth, args.cage_height,
                                   draw_labels=args.draw_labels)
                if _VIDEO_BACKEND == "opencv":
                    video_writers[cam.name].write(img)  # type: ignore
                elif _VIDEO_BACKEND == "imageio":
                    # imageio expects RGB; our render uses cv2 colors (BGR-ish), but mostly grayscale.
                    video_writers[cam.name].append_data(img[..., ::-1])  # type: ignore

    # close
    f3d.close()
    fproto.close()
    for cam in cams:
        f2d[cam.name].close()

    if args.write_video:
        if _VIDEO_BACKEND == "opencv":
            for vw in video_writers.values():
                vw.release()
        elif _VIDEO_BACKEND == "imageio":
            for vw in video_writers.values():
                vw.close()

    print(f"Done.\nOutputs in: {args.out_dir}")
    print(f"  3D coords: {path_3d}")
    print(f"  Behavior protocol (0->1): {path_protocol}")
    for cam in cams:
        print(f"  2D keypoints: {path_2d[cam.name]}")
        if args.write_video:
            print(f"  Video: {os.path.join(args.out_dir, cam.name + '.mp4')}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mouse 1.0 simulator (Option A trot).")
    p.add_argument("--mouse-graph", required=True, help="Path to mouse graph text file.")
    p.add_argument("--out-dir", default="out_mouse", help="Output directory.")

    # cage
    p.add_argument("--cage-width", type=float, default=400.0, help="Cage size in x.")
    p.add_argument("--cage-depth", type=float, default=300.0, help="Cage size in y.")
    p.add_argument("--cage-height", type=float, default=200.0, help="Cage height in z (for visualization).")

    # sim
    p.add_argument("--duration", type=float, default=20.0, help="Seconds to simulate.")
    p.add_argument("--fps", type=float, default=10.0, help="Frames per second (video + sampling).")
    p.add_argument("--seed", type=int, default=1, help="RNG seed.")

    # track dynamics
    p.add_argument("--max-speed", type=float, default=100.0, help="Max speed (units per second).")
    p.add_argument("--omega-max", type=float, default=2.2, help="Max yaw rate (rad/s).")
    p.add_argument("--speed-noise", type=float, default=60.0, help="Speed noise scale.")
    p.add_argument("--omega-noise", type=float, default=2.0, help="Yaw-rate noise scale.")
    p.add_argument("--speed-damp", type=float, default=1.5, help="Speed damping.")
    p.add_argument("--omega-damp", type=float, default=2.0, help="Yaw-rate damping.")
    p.add_argument("--wall-repulse-gain", type=float, default=2.0, help="Steering toward center near walls.")
    p.add_argument("--wall-repulse-dist", type=float, default=60.0, help="Wall proximity distance for repulsion.")

    # running gait
    p.add_argument("--stride-len-fallback", type=float, default=10.0, help="Used if head z<=0 in graph.")
    p.add_argument("--head-jitter-amp-deg", type=float, default=6.0, help="Head rigid-cluster jitter amplitude (deg).")
    p.add_argument("--head-jitter-hz", type=float, default=1.2, help="Head jitter frequency (Hz).")

    # rearing modulation
    p.add_argument("--rear-speed-scale", type=float, default=0.25, help="Speed multiplier during rearing.")
    p.add_argument("--rear-turn-noise-scale", type=float, default=1.2, help="Turn noise multiplier during rearing.")

    # camera/video
    p.add_argument("--image-width", type=int, default=960)
    p.add_argument("--image-height", type=int, default=720)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--write-video", action="store_true", help="Write mp4 videos (needs opencv or imageio).")
    p.add_argument("--draw-labels", action="store_true", help="Draw node labels into video frames.")

    p.add_argument("--export-cameras", action="store_true",
               help="Write cameras.json (intrinsics + extrinsics) to out-dir.")

    return p


def main():
    args = build_argparser().parse_args()
    simulate(args)


if __name__ == "__main__":
    main()

