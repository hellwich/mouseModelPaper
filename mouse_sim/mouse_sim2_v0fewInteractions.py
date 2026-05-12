#!/usr/bin/env python3
"""
Mouse 2.0 (two mice) — based on Mouse 1.0 Option A (trot + tightened rearing)

Adds:
- Two independently moving mice (mouse_id 0 and 1)
- Two-sphere collision proxy (front + rear) to prevent body intersection
- Optional coupled "social interaction" episode with phases: approach -> hold -> separate
- Hierarchical arbitration: hard constraints (walls + collision) > social goal > individual run/rear

Outputs:
- coords_3d.csv: per-frame 3D keypoints for both mice (includes mouse_id, behavior)
- keypoints_<cam>.csv: per-frame 2D projections for both mice (includes mouse_id, behavior)
- behavior_protocol.csv: records each per-mouse switch 0->1 (run->rear) when not in social
- interaction_protocol.csv: records social episodes (start/touch/end, initiator, min nose distance)
- Videos (optional): skeletons for both mice rendered in different colors (cam1_top, cam2_front, cam3_side)
- cameras.json (optional): scene JSON with cage + cameras (same as Mouse 1)

Behavior labels:
  0 = running
  1 = rearing
  2 = social interaction (overrides run/rear while the episode is active)

Notes / compromises (kept intentionally simple):
- Running gait uses ground-plane paw x-offsets (trot timing), not strict stance constraints.
- During social episodes, rearing is suppressed (behavior=2) and mice move with slow running.
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
    """
    x = float(v0[0]); y = float(v0[1]); z = float(v0[2])
    R = math.hypot(x, z)
    if R < 1e-12:
        return v0.copy()

    zt = max(-R, min(R, z_rel_target))

    beta0 = math.atan2(x, z)
    ang = math.acos(zt / R)

    b1 = -beta0 + ang
    b2 = -beta0 - ang

    def xprime(b: float) -> float:
        return math.cos(b)*x + math.sin(b)*z

    b = b1 if xprime(b1) < xprime(b2) else b2

    xp = math.cos(b)*x + math.sin(b)*z
    zp = -math.sin(b)*x + math.cos(b)*z
    return np.array([xp, y, zp], dtype=np.float64)

def ang_diff(a: float, b: float) -> float:
    """Return wrapped angle a-b in (-pi,pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


# ----------------------------
# Random track model (smooth heading + speed), with extra steering terms
# ----------------------------

@dataclass
class TrackState:
    x: float
    y: float
    theta: float   # yaw heading
    v: float       # speed
    omega: float   # yaw rate


def _repulse01(dist: float, dist0: float) -> float:
    """0 if dist>=dist0, 1 if dist<=0, linear in between."""
    if dist <= 1e-9:
        return 1.0
    if dist >= dist0:
        return 0.0
    return (dist0 - dist) / dist0


def step_track_steered(
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
    # optional steering:
    target_heading: Optional[float] = None,
    target_gain: float = 0.0,
    speed_target: Optional[float] = None,
    speed_gain: float = 0.0,
    avoid_heading: Optional[float] = None,
    avoid_gain: float = 0.0,
    avoid_dist: float = 0.0,
) -> TrackState:
    """
    OU-ish update for speed/omega, plus three steering terms:
      - wall: pushes toward cage center near boundaries
      - target: pulls heading toward target_heading
      - avoid: pushes heading away from avoid_heading (typically toward "away from other")
    """
    # OU-ish update
    st.v += (-speed_damp * st.v) * dt + speed_noise * math.sqrt(dt) * rng.gauss(0, 1)
    st.omega += (-omega_damp * st.omega) * dt + omega_noise * math.sqrt(dt) * rng.gauss(0, 1)

    # optionally pull speed toward speed_target (for social phases)
    if speed_target is not None and speed_gain > 0.0:
        st.v += speed_gain * (speed_target - st.v) * dt

    st.v = max(0.0, min(max_speed, st.v))
    st.omega = max(-omega_max, min(omega_max, st.omega))

    # wall steering
    cx, cy = cage_w * 0.5, cage_d * 0.5
    to_c = math.atan2(cy - st.y, cx - st.x)

    dL = st.x
    dR = cage_w - st.x
    dB = st.y
    dT = cage_d - st.y
    prox = max(_repulse01(dL, wall_repulse_dist),
               _repulse01(dR, wall_repulse_dist),
               _repulse01(dB, wall_repulse_dist),
               _repulse01(dT, wall_repulse_dist))
    steer = -wall_repulse_gain * prox * ang_diff(st.theta, to_c)

    # target steering (toward desired heading)
    if target_heading is not None and target_gain > 0.0:
        steer += -target_gain * ang_diff(st.theta, target_heading)

    # avoid steering (away from avoid_heading; apply when close)
    if avoid_heading is not None and avoid_gain > 0.0 and avoid_dist > 1e-9:
        # avoid_heading here is typically "angle to other", so "away" is avoid_heading+pi
        # We measure distance-based proximity outside and pass as avoid_dist; just steer strongly when needed.
        steer += -avoid_gain * ang_diff(st.theta, (avoid_heading + math.pi))

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
# Behaviors (per mouse)
# ----------------------------

@dataclass
class BehaviorState:
    behavior: int
    t_in_state: float
    state_duration: float
    gait_phase: float  # radians
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
    pitch_scale = 1.0 + rng.uniform(-0.08, 0.08)  # ±8%
    paw_lift = 0.30 * z_head * (1.0 + rng.uniform(-0.15, 0.15))
    paw_retract = 0.20 * z_head * (1.0 + rng.uniform(-0.15, 0.15))
    return pitch_scale, paw_lift, paw_retract


# ----------------------------
# Poses for running + rearing (same as Mouse 1)
# ----------------------------

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
    nodes = {k: v.copy() for k, v in g.nodes.items()}

    head = g.require("head")
    nose = g.require("nose_tip")
    le = g.require("left_ear_tip")
    re = g.require("right_ear_tip")
    lfp = g.require("left_front_paw")
    rfp = g.require("right_front_paw")
    lhp = g.require("left_hind_paw")
    rhp = g.require("right_hind_paw")

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

    if stride_len <= 1e-9 or v_body <= 1e-9:
        dphi = 0.0
    else:
        dphi = 2 * math.pi * (v_body / stride_len) * dt
    gait_phase = (gait_phase + dphi) % (2 * math.pi)

    def tri_wave(phase: float) -> float:
        u = (phase % (2 * math.pi)) / (2 * math.pi)
        if u < 0.5:
            return 1.0 - 4.0 * u
        else:
            return -3.0 + 4.0 * u

    def paw_x_offset(phase: float) -> float:
        return (stride_len / 4.0) * tri_wave(phase)

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

    u = 0.0 if state_duration <= 1e-9 else max(0.0, min(1.0, t_in_state / state_duration))
    h = math.sin(math.pi * u)

    a0 = g.nodes[head] - g.nodes[tail_root]
    ax = float(a0[0])
    az = float(a0[2])
    pitch_peak = math.atan2(-ax, az)
    pitch = (pitch_peak * rear_pitch_scale) * h

    pivot = g.nodes[tail_root]
    Q = roty(pitch)

    body_cluster = (head, nose, le, re, tail_tip)
    for key in body_cluster:
        v0 = g.nodes[key] - pivot
        nodes[key] = pivot + Q @ v0

    for key in (lhp, rhp):
        p0 = g.nodes[key]
        nodes[key] = np.array([p0[0], p0[1], 0.0], dtype=np.float64)

    head_pos = nodes[head]
    paw_z_abs = rear_paw_lift * h

    for paw in (lfp, rfp):
        v0 = g.nodes[paw] - g.nodes[head]
        z_rel_target = paw_z_abs - float(head_pos[2])
        v_rot = rotate_leg_vec_about_y_to_reach_z(v0, z_rel_target)
        nodes[paw] = head_pos + v_rot

    return nodes


# ----------------------------
# Cameras and projection (unchanged)
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
    eye1 = np.array([cage_w/2, cage_d/2, cage_h/2 + dist], dtype=np.float64)
    R1, t1 = look_at(eye1, C, np.array([0, 1, 0], dtype=np.float64))
    cams.append(Camera("cam1_top", image_w, image_h, fx, fy, cx, cy, R1, t1, eye1))
    eye2 = np.array([cage_w/2, cage_d/2 - dist, cage_h/2 + 0.25*cage_h], dtype=np.float64)
    R2, t2 = look_at(eye2, C, np.array([0, 0, 1], dtype=np.float64))
    cams.append(Camera("cam2_front", image_w, image_h, fx, fy, cx, cy, R2, t2, eye2))
    eye3 = np.array([cage_w/2 - dist, cage_d/2, cage_h/2 + 0.25*cage_h], dtype=np.float64)
    R3, t3 = look_at(eye3, C, np.array([0, 0, 1], dtype=np.float64))
    cams.append(Camera("cam3_side", image_w, image_h, fx, fy, cx, cy, R3, t3, eye3))
    return cams


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
            "R": cam.R.tolist(),
            "t": cam.t.tolist(),
            "pos_world": cam.pos_world.tolist(),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# Two-sphere collision proxy
# ----------------------------

@dataclass
class BodyProxy2S:
    cF_local: np.ndarray  # (2,) xy
    cR_local: np.ndarray  # (2,) xy
    rF: float
    rR: float

    @property
    def r_max(self) -> float:
        return max(self.rF, self.rR)


def build_body_proxy_2sphere(g: MouseGraph, front_center_mode: str = "head_nose_mid") -> BodyProxy2S:
    head = g.require("head")
    nose = g.require("nose_tip")
    tail_root = g.require("tail_root")

    if front_center_mode == "head":
        cF = g.nodes[head]
    else:
        cF = 0.5 * (g.nodes[head] + g.nodes[nose])
    cR = g.nodes[tail_root]

    cF_xy = np.array([float(cF[0]), float(cF[1])], dtype=np.float64)
    cR_xy = np.array([float(cR[0]), float(cR[1])], dtype=np.float64)

    x_split = 0.5 * (float(cF[0]) + float(cR[0]))
    front_nodes = [p for p in g.nodes.values() if float(p[0]) >= x_split]
    rear_nodes = [p for p in g.nodes.values() if float(p[0]) < x_split]
    if not front_nodes:
        front_nodes = list(g.nodes.values())
    if not rear_nodes:
        rear_nodes = list(g.nodes.values())

    def max_r_xy(nodes: List[np.ndarray], cxy: np.ndarray) -> float:
        r = 0.0
        for p in nodes:
            dx = float(p[0]) - float(cxy[0])
            dy = float(p[1]) - float(cxy[1])
            r = max(r, math.hypot(dx, dy))
        return r

    rF = max_r_xy(front_nodes, cF_xy)
    rR = max_r_xy(rear_nodes, cR_xy)

    return BodyProxy2S(cF_local=cF_xy, cR_local=cR_xy, rF=float(rF), rR=float(rR))


def sphere_centers_xy(st: TrackState, proxy: BodyProxy2S) -> Tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(st.theta), math.sin(st.theta)
    R2 = np.array([[c, -s],
                   [s,  c]], dtype=np.float64)
    p = np.array([st.x, st.y], dtype=np.float64)
    cF = p + R2 @ proxy.cF_local
    cR = p + R2 @ proxy.cR_local
    return cF, cR


def project_constraints_two_mice(
    st0: TrackState,
    st1: TrackState,
    proxy: BodyProxy2S,
    cage_w: float,
    cage_d: float,
    margin: float,
    dominance0: float,
    dominance1: float,
    n_iter: int = 6,
) -> Tuple[TrackState, TrackState]:
    """
    Position-based projection in XY:
      - keep both sphere centers inside cage rectangle with margin
      - keep all sphere-pairs separated by r_i+r_j+margin
    Dominance in [0,1]: higher => moves less when resolving overlaps.
    """
    # weights for moving each mouse during mouse-mouse resolution:
    inv0 = 1.0 - float(dominance0)
    inv1 = 1.0 - float(dominance1)
    w0 = inv0 / (inv0 + inv1 + 1e-12)
    w1 = inv1 / (inv0 + inv1 + 1e-12)

    for _ in range(max(1, n_iter)):
        # --- wall constraints (apply using worst violation across both spheres) ---
        for st in (st0, st1):
            cF, cR = sphere_centers_xy(st, proxy)
            # x min/max
            xmin_req = max(proxy.rF, proxy.rR) + margin  # conservative bound
            xmax_req = cage_w - xmin_req
            ymin_req = max(proxy.rF, proxy.rR) + margin
            ymax_req = cage_d - ymin_req

            # but we can do tighter using each sphere (still cheap):
            for (cxy, r) in ((cF, proxy.rF), (cR, proxy.rR)):
                if cxy[0] < r + margin:
                    st.x += float((r + margin) - cxy[0])
                if cxy[0] > cage_w - (r + margin):
                    st.x -= float(cxy[0] - (cage_w - (r + margin)))
                if cxy[1] < r + margin:
                    st.y += float((r + margin) - cxy[1])
                if cxy[1] > cage_d - (r + margin):
                    st.y -= float(cxy[1] - (cage_d - (r + margin)))

        # --- mouse-mouse constraints ---
        cF0, cR0 = sphere_centers_xy(st0, proxy)
        cF1, cR1 = sphere_centers_xy(st1, proxy)

        pairs = [
            (cF0, proxy.rF, cF1, proxy.rF),
            (cF0, proxy.rF, cR1, proxy.rR),
            (cR0, proxy.rR, cF1, proxy.rF),
            (cR0, proxy.rR, cR1, proxy.rR),
        ]
        any_overlap = False
        for a, ra, b, rb in pairs:
            dvec = b - a
            d = float(np.linalg.norm(dvec))
            dmin = float(ra + rb + margin)
            if d < dmin:
                any_overlap = True
                if d < 1e-9:
                    # arbitrary push direction
                    n = np.array([1.0, 0.0], dtype=np.float64)
                else:
                    n = dvec / d
                pen = dmin - d
                st0.x -= float(w0 * pen * n[0])
                st0.y -= float(w0 * pen * n[1])
                st1.x += float(w1 * pen * n[0])
                st1.y += float(w1 * pen * n[1])

        if not any_overlap:
            break

    return st0, st1


# ----------------------------
# Social interaction (pair-level FSM)
# ----------------------------

class SocialMode:
    NONE = 0
    APPROACH = 1
    HOLD = 2
    SEPARATE = 3


@dataclass
class SocialState:
    mode: int
    t_in: float
    next_start_t: float
    initiator: int
    meet_xy: np.ndarray
    start_frame: int
    start_time: float
    touch_frame: int
    touch_time: float
    min_nose_dist: float


def schedule_next_social(t_now: float, rate_per_min: float, cooldown: float, rng: random.Random) -> float:
    if rate_per_min <= 1e-12:
        return float("inf")
    lam = rate_per_min / 60.0
    return t_now + cooldown + rng.expovariate(lam)


def clamp_meeting_point(mid: np.ndarray, cage_w: float, cage_d: float, pad: float) -> np.ndarray:
    x = min(max(float(mid[0]), pad), float(cage_w) - pad)
    y = min(max(float(mid[1]), pad), float(cage_d) - pad)
    return np.array([x, y], dtype=np.float64)


# ----------------------------
# Rendering (2 mice)
# ----------------------------

def render_frame_two(
    cam: Camera,
    g: MouseGraph,
    world_nodes_by_mouse: List[Dict[str, np.ndarray]],
    cage_w: float,
    cage_d: float,
    cage_h: float,
    draw_labels: bool = False
) -> np.ndarray:
    img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    if cv2 is None:
        return img

    # cage outline (same as Mouse 1)
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

    # mouse colors (BGR)
    colors = [(255, 255, 255), (80, 220, 80)]
    node_cols = [(255, 255, 255), (120, 255, 120)]

    for mi, world_nodes in enumerate(world_nodes_by_mouse):
        col = colors[mi % len(colors)]
        coln = node_cols[mi % len(node_cols)]

        for a, b in g.edges:
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


# ----------------------------
# Main simulation (two mice)
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
    stride_len *= float(args.stride_scale)

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
    path_behavior = os.path.join(args.out_dir, "behavior_protocol.csv")
    path_social = os.path.join(args.out_dir, "interaction_protocol.csv")
    path_2d = {cam.name: os.path.join(args.out_dir, f"keypoints_{cam.name}.csv") for cam in cams}

    f3d = open(path_3d, "w", newline="", encoding="utf-8")
    w3d = csv.writer(f3d)
    w3d.writerow(["frame", "time", "mouse_id", "behavior", "node", "x", "y", "z"])

    fbeh = open(path_behavior, "w", newline="", encoding="utf-8")
    wbeh = csv.writer(fbeh)
    wbeh.writerow(["frame", "time", "mouse_id", "from_behavior", "to_behavior", "rearing_duration",
                   "rear_pitch_scale", "rear_paw_lift", "rear_paw_retract"])

    fsoc = open(path_social, "w", newline="", encoding="utf-8")
    wsoc = csv.writer(fsoc)
    wsoc.writerow(["start_frame", "start_time", "touch_frame", "touch_time",
                   "end_frame", "end_time", "initiator", "end_reason", "min_nose_dist"])

    f2d = {}
    w2d = {}
    for cam in cams:
        f2d[cam.name] = open(path_2d[cam.name], "w", newline="", encoding="utf-8")
        w2d[cam.name] = csv.writer(f2d[cam.name])
        w2d[cam.name].writerow(["frame", "time", "mouse_id", "behavior", "node", "u", "v", "visible"])

    # build collision proxy
    proxy = build_body_proxy_2sphere(g, front_center_mode=args.proxy_front_center)
    proxy_scale = float(args.proxy_scale)
    proxy = BodyProxy2S(proxy.cF_local, proxy.cR_local, proxy.rF*proxy_scale, proxy.rR*proxy_scale)

    # dominance
    dom0 = max(0.0, min(1.0, rng.random()))
    dom1 = max(0.0, min(1.0, rng.random()))

    # init track states (place with separation along x, clamp inside)
    sep = float(args.init_separation)
    x0 = args.cage_width * 0.5 - 0.5 * sep
    x1 = args.cage_width * 0.5 + 0.5 * sep
    y0 = args.cage_depth * 0.5
    y1 = args.cage_depth * 0.5

    st = [
        TrackState(x=x0, y=y0, theta=rng.uniform(-math.pi, math.pi), v=rng.uniform(0.0, args.max_speed), omega=0.0),
        TrackState(x=x1, y=y1, theta=rng.uniform(-math.pi, math.pi), v=rng.uniform(0.0, args.max_speed), omega=0.0),
    ]
    # push inside walls using projection once
    st[0], st[1] = project_constraints_two_mice(
        st[0], st[1], proxy, args.cage_width, args.cage_depth, args.collision_margin, dom0, dom1, n_iter=10
    )

    # init behavior states (running)
    beh: List[BehaviorState] = []
    for _ in range(2):
        pitch_scale, paw_lift, paw_retract = sample_rearing_params(z_head, rng)
        beh.append(BehaviorState(
            behavior=0,
            t_in_state=0.0,
            state_duration=sample_duration(0, rng),
            gait_phase=rng.uniform(0.0, 2 * math.pi),
            rear_pitch_scale=pitch_scale,
            rear_paw_lift=paw_lift,
            rear_paw_retract=paw_retract,
        ))

    dt = 1.0 / args.fps
    n_frames = int(round(args.duration * args.fps))

    # social state
    ss = SocialState(
        mode=SocialMode.NONE,
        t_in=0.0,
        next_start_t=schedule_next_social(0.0, args.social_rate_per_min, args.social_cooldown, rng),
        initiator=0,
        meet_xy=np.array([args.cage_width/2, args.cage_depth/2], dtype=np.float64),
        start_frame=-1,
        start_time=0.0,
        touch_frame=-1,
        touch_time=0.0,
        min_nose_dist=float("inf"),
    )

    def nose_pos_world(stm: TrackState) -> np.ndarray:
        # approximate using rest nose position rotated by yaw only
        nose = g.nodes[_norm_label("nose_tip")]
        Rm = rotz(stm.theta)
        Pm = np.array([stm.x, stm.y, 0.0], dtype=np.float64)
        return Pm + Rm @ nose

    def center_xy(stm: TrackState) -> np.ndarray:
        return np.array([stm.x, stm.y], dtype=np.float64)

    # Optional: precompute avoidance distance based on proxy
    avoid_dist = float(args.avoid_dist)
    if avoid_dist <= 0.0:
        avoid_dist = 2.2 * (proxy.rF + proxy.rR + args.collision_margin)

    # per-frame
    for frame in range(n_frames):
        t = frame * dt

        # --- SOCIAL FSM: possibly start or progress ---
        if ss.mode == SocialMode.NONE and t >= ss.next_start_t:
            # start a new episode if mice aren't already too close
            dcent = float(np.linalg.norm(center_xy(st[0]) - center_xy(st[1])))
            if dcent >= float(args.social_min_start_dist):
                ss.mode = SocialMode.APPROACH
                ss.t_in = 0.0
                # choose initiator (dominance-weighted)
                p0 = dom0 / (dom0 + dom1 + 1e-12)
                ss.initiator = 0 if (rng.random() < p0) else 1
                mid = 0.5 * (center_xy(st[0]) + center_xy(st[1]))
                pad = proxy.r_max + args.collision_margin + float(args.social_wall_pad)
                ss.meet_xy = clamp_meeting_point(mid, args.cage_width, args.cage_depth, pad)
                ss.start_frame = frame
                ss.start_time = t
                ss.touch_frame = -1
                ss.touch_time = 0.0
                ss.min_nose_dist = float("inf")
            # always reschedule next start (cooldown prevents rapid retries)
            ss.next_start_t = schedule_next_social(t, args.social_rate_per_min, args.social_cooldown, rng)

        # whether social overrides per-mouse behavior
        social_active = ss.mode != SocialMode.NONE

        # --- per-mouse behavior timers (only if NOT social) ---
        for mi in (0, 1):
            if social_active:
                continue  # freeze run/rear state during social episode
            beh[mi].t_in_state += dt
            if beh[mi].t_in_state >= beh[mi].state_duration:
                prev = beh[mi].behavior
                newb = next_behavior(beh[mi].behavior, rng)
                beh[mi].behavior = newb
                beh[mi].t_in_state = 0.0
                beh[mi].state_duration = sample_duration(beh[mi].behavior, rng)
                if prev == 0 and newb == 1:
                    beh[mi].rear_pitch_scale, beh[mi].rear_paw_lift, beh[mi].rear_paw_retract = sample_rearing_params(z_head, rng)
                    wbeh.writerow([
                        frame, f"{t:.6f}", mi, prev, newb, f"{beh[mi].state_duration:.6f}",
                        f"{beh[mi].rear_pitch_scale:.6f}", f"{beh[mi].rear_paw_lift:.6f}", f"{beh[mi].rear_paw_retract:.6f}"
                    ])

        # --- determine steering targets and speed targets (hierarchical) ---
        # avoid other always (softly), but social target can override heading and speed.
        target_heading = [None, None]
        speed_target = [None, None]
        target_gain = [0.0, 0.0]
        speed_gain = [0.0, 0.0]

        if social_active:
            ss.t_in += dt
            # compute nose distance for monitoring / transitions later (after we compute final pose too)
            n0 = nose_pos_world(st[0])
            n1 = nose_pos_world(st[1])
            ss.min_nose_dist = min(ss.min_nose_dist, float(np.linalg.norm(n0 - n1)))

            if ss.mode == SocialMode.APPROACH:
                # meeting behavior: initiator moves more, responder yields
                for mi in (0, 1):
                    if mi == ss.initiator:
                        goal = ss.meet_xy
                        v_tgt = float(args.social_approach_speed)
                        g_gain = float(args.social_target_gain)
                    else:
                        # face partner and drift toward meeting point slowly
                        goal = 0.7 * ss.meet_xy + 0.3 * center_xy(st[ss.initiator])
                        v_tgt = float(args.social_approach_speed) * float(args.social_responder_speed_scale)
                        g_gain = float(args.social_target_gain) * 0.8
                    vec = goal - center_xy(st[mi])
                    target_heading[mi] = math.atan2(float(vec[1]), float(vec[0]))
                    speed_target[mi] = v_tgt
                    target_gain[mi] = g_gain
                    speed_gain[mi] = float(args.social_speed_gain)

                # transition to HOLD when nose distance small enough
                if float(np.linalg.norm(n0 - n1)) <= float(args.social_touch_dist):
                    ss.mode = SocialMode.HOLD
                    ss.t_in = 0.0
                    ss.touch_frame = frame
                    ss.touch_time = t

                # timeout abort
                if ss.t_in >= float(args.social_approach_max):
                    # end episode
                    wsoc.writerow([ss.start_frame, f"{ss.start_time:.6f}",
                                   ss.touch_frame, f"{ss.touch_time:.6f}" if ss.touch_frame >= 0 else "",
                                   frame, f"{t:.6f}", ss.initiator, "approach_timeout", f"{ss.min_nose_dist:.6f}"])
                    ss.mode = SocialMode.NONE
                    ss.t_in = 0.0

            elif ss.mode == SocialMode.HOLD:
                for mi in (0, 1):
                    other = 1 - mi
                    vec = center_xy(st[other]) - center_xy(st[mi])
                    target_heading[mi] = math.atan2(float(vec[1]), float(vec[0]))
                    speed_target[mi] = float(args.social_hold_speed)
                    target_gain[mi] = float(args.social_target_gain) * 0.6
                    speed_gain[mi] = float(args.social_speed_gain)

                if ss.t_in >= float(args.social_hold_time):
                    ss.mode = SocialMode.SEPARATE
                    ss.t_in = 0.0

            elif ss.mode == SocialMode.SEPARATE:
                for mi in (0, 1):
                    other = 1 - mi
                    vec = center_xy(st[mi]) - center_xy(st[other])  # away
                    target_heading[mi] = math.atan2(float(vec[1]), float(vec[0]))
                    speed_target[mi] = float(args.social_separate_speed)
                    target_gain[mi] = float(args.social_target_gain) * 0.8
                    speed_gain[mi] = float(args.social_speed_gain)

                dcent = float(np.linalg.norm(center_xy(st[0]) - center_xy(st[1])))
                if dcent >= float(args.social_separate_dist):
                    wsoc.writerow([ss.start_frame, f"{ss.start_time:.6f}",
                                   ss.touch_frame, f"{ss.touch_time:.6f}" if ss.touch_frame >= 0 else "",
                                   frame, f"{t:.6f}", ss.initiator, "complete", f"{ss.min_nose_dist:.6f}"])
                    ss.mode = SocialMode.NONE
                    ss.t_in = 0.0

                if ss.t_in >= float(args.social_separate_max):
                    wsoc.writerow([ss.start_frame, f"{ss.start_time:.6f}",
                                   ss.touch_frame, f"{ss.touch_time:.6f}" if ss.touch_frame >= 0 else "",
                                   frame, f"{t:.6f}", ss.initiator, "separate_timeout", f"{ss.min_nose_dist:.6f}"])
                    ss.mode = SocialMode.NONE
                    ss.t_in = 0.0

        # --- step both tracks (desired -> then constrained) ---
        # compute avoid headings (angle to other mouse)
        vec01 = center_xy(st[1]) - center_xy(st[0])
        d01 = float(np.linalg.norm(vec01))
        ang01 = math.atan2(float(vec01[1]), float(vec01[0])) if d01 > 1e-9 else 0.0

        for mi in (0, 1):
            # scale speed / noise by per-mouse base behavior unless social overrides
            if social_active:
                max_speed = float(args.max_speed) * float(args.social_speed_scale)
                omega_noise = float(args.omega_noise) * float(args.social_turn_noise_scale)
                behavior_label = 2
            else:
                if beh[mi].behavior == 1:
                    max_speed = float(args.max_speed) * float(args.rear_speed_scale)
                    omega_noise = float(args.omega_noise) * float(args.rear_turn_noise_scale)
                    behavior_label = 1
                else:
                    max_speed = float(args.max_speed)
                    omega_noise = float(args.omega_noise)
                    behavior_label = 0

            # avoid settings
            if d01 < avoid_dist:
                avoid_gain = float(args.avoid_gain) * _repulse01(d01, avoid_dist)
            else:
                avoid_gain = 0.0

            avoid_heading = (ang01 if mi == 0 else (ang01 + math.pi))

            st[mi] = step_track_steered(
                st=st[mi], dt=dt,
                cage_w=args.cage_width, cage_d=args.cage_depth,
                max_speed=max_speed,
                omega_max=args.omega_max,
                speed_noise=args.speed_noise,
                omega_noise=omega_noise,
                speed_damp=args.speed_damp,
                omega_damp=args.omega_damp,
                wall_repulse_gain=args.wall_repulse_gain,
                wall_repulse_dist=args.wall_repulse_dist,
                rng=rng,
                target_heading=target_heading[mi],
                target_gain=target_gain[mi],
                speed_target=speed_target[mi],
                speed_gain=speed_gain[mi],
                avoid_heading=avoid_heading,
                avoid_gain=avoid_gain,
                avoid_dist=avoid_dist,
            )

        # hard constraints
        st[0], st[1] = project_constraints_two_mice(
            st[0], st[1], proxy, args.cage_width, args.cage_depth,
            args.collision_margin, dom0, dom1, n_iter=args.collision_iters
        )

        # --- local poses and world nodes for both mice ---
        world_nodes_by_mouse: List[Dict[str, np.ndarray]] = []
        behavior_by_mouse: List[int] = []

        for mi in (0, 1):
            if social_active:
                behavior_label = 2
                local_nodes, beh[mi].gait_phase = pose_running_option_a(
                    g=g, dt=dt, v_body=st[mi].v,
                    gait_phase=beh[mi].gait_phase,
                    stride_len=stride_len,
                    head_jitter_amp_deg=args.head_jitter_amp_deg,
                    head_jitter_hz=args.head_jitter_hz,
                    t_global=t + 10.0 * mi,
                )
            else:
                if beh[mi].behavior == 0:
                    behavior_label = 0
                    local_nodes, beh[mi].gait_phase = pose_running_option_a(
                        g=g, dt=dt, v_body=st[mi].v,
                        gait_phase=beh[mi].gait_phase,
                        stride_len=stride_len,
                        head_jitter_amp_deg=args.head_jitter_amp_deg,
                        head_jitter_hz=args.head_jitter_hz,
                        t_global=t + 10.0 * mi,
                    )
                else:
                    behavior_label = 1
                    local_nodes = pose_rearing_tight(
                        g=g,
                        t_in_state=beh[mi].t_in_state,
                        state_duration=beh[mi].state_duration,
                        rear_pitch_scale=beh[mi].rear_pitch_scale,
                        rear_paw_lift=beh[mi].rear_paw_lift,
                        rear_paw_retract=beh[mi].rear_paw_retract,
                    )

            Rm = rotz(st[mi].theta)
            Pm = np.array([st[mi].x, st[mi].y, 0.0], dtype=np.float64)
            world_nodes = {k: (Pm + Rm @ v) for k, v in local_nodes.items()}

            world_nodes_by_mouse.append(world_nodes)
            behavior_by_mouse.append(int(behavior_label))

            # write 3D
            for node in g.node_order:
                Pw = world_nodes[node]
                w3d.writerow([frame, f"{t:.6f}", mi, behavior_label, node,
                              f"{Pw[0]:.6f}", f"{Pw[1]:.6f}", f"{Pw[2]:.6f}"])

        # update social monitoring using actual world nose positions (more accurate than yaw-only approx)
        if social_active and ss.mode != SocialMode.NONE:
            n0w = world_nodes_by_mouse[0][_norm_label("nose_tip")]
            n1w = world_nodes_by_mouse[1][_norm_label("nose_tip")]
            dist_nose = float(np.linalg.norm(n0w - n1w))
            ss.min_nose_dist = min(ss.min_nose_dist, dist_nose)

            # post-step touch detection (avoids 1-frame lag)
            if ss.mode == SocialMode.APPROACH and dist_nose <= float(args.social_touch_dist):
                ss.mode = SocialMode.HOLD
                ss.t_in = 0.0
                if ss.touch_frame < 0:
                    ss.touch_frame = frame
                    ss.touch_time = t

        # per-camera 2D + video
        for cam in cams:
            for mi in (0, 1):
                world_nodes = world_nodes_by_mouse[mi]
                b = behavior_by_mouse[mi]
                for node in g.node_order:
                    u, v, vis = cam.project(world_nodes[node])
                    w2d[cam.name].writerow([frame, f"{t:.6f}", mi, b, node,
                                            f"{u:.6f}", f"{v:.6f}", int(vis)])

            if args.write_video:
                img = render_frame_two(cam, g, world_nodes_by_mouse,
                                       args.cage_width, args.cage_depth, args.cage_height,
                                       draw_labels=args.draw_labels)
                if _VIDEO_BACKEND == "opencv":
                    video_writers[cam.name].write(img)  # type: ignore
                elif _VIDEO_BACKEND == "imageio":
                    video_writers[cam.name].append_data(img[..., ::-1])  # type: ignore

    # close
    f3d.close()
    fbeh.close()
    fsoc.close()
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
    print(f"  Behavior protocol (0->1): {path_behavior}")
    print(f"  Social protocol: {path_social}")
    for cam in cams:
        print(f"  2D keypoints: {path_2d[cam.name]}")
        if args.write_video:
            print(f"  Video: {os.path.join(args.out_dir, cam.name + '.mp4')}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mouse 2.0 simulator (two mice + social interaction).")
    p.add_argument("--mouse-graph", required=True, help="Path to mouse graph text file.")
    p.add_argument("--out-dir", default="out_mouse2", help="Output directory.")

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
    p.add_argument("--stride-scale", type=float, default=1.0, help="Multiply stride length (z_head/3 * stride_scale).")
    p.add_argument("--head-jitter-amp-deg", type=float, default=6.0, help="Head rigid-cluster jitter amplitude (deg).")
    p.add_argument("--head-jitter-hz", type=float, default=1.2, help="Head jitter frequency (Hz).")

    # rearing modulation
    p.add_argument("--rear-speed-scale", type=float, default=0.25, help="Speed multiplier during rearing.")
    p.add_argument("--rear-turn-noise-scale", type=float, default=1.2, help="Turn noise multiplier during rearing.")

    # collision proxy / avoidance
    p.add_argument("--proxy-front-center", default="head_nose_mid", choices=["head_nose_mid", "head"],
                   help="How to place the front sphere center in local coords.")
    p.add_argument("--proxy-scale", type=float, default=0.85,
                   help="Scale both proxy radii (<=1 makes interaction easier).")
    p.add_argument("--collision-margin", type=float, default=5.0,
                   help="Extra clearance added to sphere-sphere and wall constraints.")
    p.add_argument("--collision-iters", type=int, default=6,
                   help="Projection iterations for constraints per frame.")
    p.add_argument("--init-separation", type=float, default=120.0,
                   help="Initial separation between the two mice along x.")
    p.add_argument("--avoid-gain", type=float, default=2.0, help="Steering away from the other mouse (soft).")
    p.add_argument("--avoid-dist", type=float, default=0.0,
                   help="Distance at which avoidance starts (0 => auto based on proxy).")

    # social interaction
    p.add_argument("--social-rate-per-min", type=float, default=3.0,
                   help="Expected social episodes per minute (0 disables).")
    p.add_argument("--social-cooldown", type=float, default=3.0,
                   help="Cooldown (s) added after scheduling an episode.")
    p.add_argument("--social-min-start-dist", type=float, default=80.0,
                   help="Start social only if center distance >= this (prevents instant episodes).")
    p.add_argument("--social-wall-pad", type=float, default=20.0,
                   help="Extra pad (beyond proxy radius) when clamping the meeting point away from walls.")

    p.add_argument("--social-touch-dist", type=float, default=18.0, help="Nose-to-nose distance threshold.")
    p.add_argument("--social-separate-dist", type=float, default=180.0, help="Center distance to finish separation.")
    p.add_argument("--social-approach-speed", type=float, default=50.0)
    p.add_argument("--social-responder-speed-scale", type=float, default=0.4)
    p.add_argument("--social-hold-speed", type=float, default=8.0)
    p.add_argument("--social-separate-speed", type=float, default=60.0)
    p.add_argument("--social-speed-gain", type=float, default=3.0,
                   help="How quickly speed tracks social targets (higher => more immediate).")
    p.add_argument("--social-target-gain", type=float, default=2.5,
                   help="How strongly heading is pulled toward social targets.")
    p.add_argument("--social-speed-scale", type=float, default=0.8,
                   help="Max-speed multiplier while social is active.")
    p.add_argument("--social-turn-noise-scale", type=float, default=0.8,
                   help="Turn noise multiplier while social is active.")
    p.add_argument("--social-approach-max", type=float, default=6.0, help="Approach timeout (s).")
    p.add_argument("--social-hold-time", type=float, default=1.0, help="Hold duration (s).")
    p.add_argument("--social-separate-max", type=float, default=4.0, help="Separate timeout (s).")

    # camera/video
    p.add_argument("--image-width", type=int, default=960)
    p.add_argument("--image-height", type=int, default=720)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--write-video", action="store_true", help="Write mp4 videos (needs opencv or imageio).")
    p.add_argument("--draw-labels", action="store_true", help="Draw node labels into video frames.")
    p.add_argument("--export-cameras", action="store_true",
                   help="Write cameras.json (cage + intrinsics + extrinsics) to out-dir.")

    return p


def main():
    args = build_argparser().parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
