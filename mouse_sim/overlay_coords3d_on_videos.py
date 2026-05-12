#!/usr/bin/env python3
"""
Overlay mouse skeletons from coords_3d.csv onto three camera videos using cameras.json,
and optionally write a joined mosaic video.

This is a simple standalone combination of the projection/camera handling from the
visualization script and the skeleton template conventions used by mouse_sim2.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------
# Camera model
# -----------------------------
@dataclass(frozen=True)
class PinholeCamera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # 3x3 world->cam
    t: np.ndarray  # 3 world->cam
    dist: Optional[Sequence[float]] = None

    def project(self, xw: np.ndarray) -> Optional[Tuple[float, float]]:
        xc = self.R @ xw.reshape(3,) + self.t.reshape(3,)
        z = float(xc[2])
        if z <= 1e-9:
            return None
        u = self.fx * (float(xc[0]) / z) + self.cx
        v = self.fy * (float(xc[1]) / z) + self.cy
        if 0.0 <= u < self.width and 0.0 <= v < self.height:
            return (u, v)
        return None


def load_cameras_json(path: str) -> Dict[str, PinholeCamera]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        cam_list = data
    elif isinstance(data, dict):
        if "cameras" in data:
            cam_list = data["cameras"]
        elif "camera" in data:
            cam_list = data["camera"]
        elif all(isinstance(v, dict) for v in data.values()):
            cam_list = [dict({"name": k}, **v) for k, v in data.items()]
        else:
            raise ValueError(f"Unrecognized cameras JSON structure in {path}")
    else:
        raise ValueError(f"Unrecognized cameras JSON structure in {path}")

    cams: Dict[str, PinholeCamera] = {}
    for i, c in enumerate(cam_list):
        if not isinstance(c, dict):
            raise ValueError(f"Camera entry #{i} must be an object")
        name = str(c.get("name", f"cam{i}"))

        # Intrinsics
        fx = fy = cx = cy = None
        dist = None
        width = int(c.get("width", 0))
        height = int(c.get("height", 0))
        if "intrinsics" in c and isinstance(c["intrinsics"], dict):
            intr = c["intrinsics"]
            if "K" in intr:
                K = np.asarray(intr["K"], dtype=float).reshape(3, 3)
                fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            else:
                fx = float(intr.get("fx", intr.get("f_x", intr.get("f", 0.0))))
                fy = float(intr.get("fy", intr.get("f_y", fx)))
                cx = float(intr.get("cx", intr.get("c_x", 0.0)))
                cy = float(intr.get("cy", intr.get("c_y", 0.0)))
                width = int(intr.get("width", width))
                height = int(intr.get("height", height))
            dist = intr.get("dist", intr.get("distortion", intr.get("k", None)))
        elif all(k in c for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])
            dist = c.get("dist", c.get("distortion", None))
        elif "K" in c:
            K = np.asarray(c["K"], dtype=float).reshape(3, 3)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            dist = c.get("dist", c.get("distortion", None))

        if fx is None or fy is None or cx is None or cy is None:
            raise KeyError(f"Camera '{name}' is missing intrinsics")

        # Extrinsics
        if "extrinsics" in c and isinstance(c["extrinsics"], dict):
            ext = c["extrinsics"]
            R = ext.get("R", ext.get("rotation", None))
            t = ext.get("t", ext.get("translation", None))
        else:
            R = c.get("R", c.get("rotation", None))
            t = c.get("t", c.get("translation", None))
        if R is None or t is None:
            raise KeyError(f"Camera '{name}' is missing extrinsics")

        cams[name] = PinholeCamera(
            name=name,
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            R=np.asarray(R, dtype=float).reshape(3, 3),
            t=np.asarray(t, dtype=float).reshape(3),
            dist=dist,
        )
    return cams


# -----------------------------
# Template / colors
# -----------------------------
def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("BGR must be 'B,G,R'")
    vals = []
    for p in parts:
        v = int(float(p))
        vals.append(max(0, min(255, v)))
    return (vals[0], vals[1], vals[2])


def load_template_edges(template_path: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    in_edges = False
    with open(template_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper() == "[EDGES]":
                in_edges = True
                continue
            if line.startswith("[") and line.endswith("]") and line.upper() != "[EDGES]":
                in_edges = False
                continue
            if not in_edges:
                continue
            parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            if len(parts) >= 2:
                edges.append((parts[0].strip().lower(), parts[1].strip().lower()))
    if not edges:
        raise ValueError(f"No edges found in [EDGES] section of {template_path}")
    return edges


def _draw_frame_number(frame_bgr: np.ndarray, frame_index: int, *, font_scale: float = 1.0) -> None:
    txt = str(int(frame_index))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    margin = int(10 * font_scale)
    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
    x, y = margin, margin + th
    cv2.rectangle(
        frame_bgr,
        (x - margin, y - th - margin),
        (x + tw + margin, y + baseline + margin),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(frame_bgr, txt, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# -----------------------------
# coords_3d loading and projection
# -----------------------------
def norm_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def load_coords_3d(path: str) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    Returns nested dict:
      frame -> mouse_id -> node -> np.array([x,y,z])
    Supports coords_3d.csv with columns frame,time,mouse_id,behavior,node,x,y,z.
    """
    out: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"frame", "mouse_id", "node", "x", "y", "z"}
        have = set(r.fieldnames or [])
        if not required.issubset(have):
            raise ValueError(f"{path} missing required columns. Have: {r.fieldnames}")
        for row in r:
            frame = int(row["frame"])
            mid = int(row["mouse_id"])
            node = norm_label(row["node"])
            xyz = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ], dtype=np.float64)
            out.setdefault(frame, {}).setdefault(mid, {})[node] = xyz
    if not out:
        raise ValueError(f"No rows in {path}")
    return out


def build_projected_index(
    coords_by_frame: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    cameras: Dict[str, PinholeCamera],
    camera_names: Sequence[str],
) -> Dict[str, Dict[int, Dict[int, Dict[str, Tuple[int, int]]]]]:
    """
    Returns:
      cam_name -> frame -> mouse_id -> node -> (u,v)
    Only visible/in-frame points are stored.
    """
    out: Dict[str, Dict[int, Dict[int, Dict[str, Tuple[int, int]]]]] = {c: {} for c in camera_names}
    for frame, mice in coords_by_frame.items():
        for cam_name in camera_names:
            cam = cameras[cam_name]
            frame_dict: Dict[int, Dict[str, Tuple[int, int]]] = {}
            for mid, nodes in mice.items():
                nd: Dict[str, Tuple[int, int]] = {}
                for node, xyz in nodes.items():
                    uv = cam.project(xyz)
                    if uv is None:
                        continue
                    nd[node] = (int(round(uv[0])), int(round(uv[1])))
                if nd:
                    frame_dict[mid] = nd
            if frame_dict:
                out[cam_name][frame] = frame_dict
    return out


# -----------------------------
# Drawing / writing video
# -----------------------------
def draw_projected_skeletons(
    frame_bgr: np.ndarray,
    projected_for_frame: Optional[Dict[int, Dict[str, Tuple[int, int]]]],
    edges: List[Tuple[str, str]],
    mouse_colors: Dict[int, Tuple[int, int, int]],
    point_radius: int,
    line_thickness: int,
    draw_labels: bool,
) -> np.ndarray:
    out = frame_bgr.copy()
    if not projected_for_frame:
        return out

    for mid in sorted(projected_for_frame.keys()):
        pts = projected_for_frame[mid]
        col = mouse_colors.get(mid, (255, 255, 255))

        for a, b in edges:
            if a in pts and b in pts:
                cv2.line(out, pts[a], pts[b], col, line_thickness, cv2.LINE_AA)

        for node, (u, v) in pts.items():
            cv2.circle(out, (u, v), point_radius, col, -1, cv2.LINE_AA)
            if draw_labels:
                cv2.putText(
                    out,
                    f"{mid}:{node}",
                    (u + 4, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    col,
                    1,
                    cv2.LINE_AA,
                )
    return out


def overlay_video(
    in_path: str,
    out_path: str,
    projected_by_frame: Dict[int, Dict[int, Dict[str, Tuple[int, int]]]],
    edges: List[Tuple[str, str]],
    mouse_colors: Dict[int, Tuple[int, int, int]],
    point_radius: int,
    line_thickness: int,
    draw_labels: bool,
    start_frame: int,
    end_frame: Optional[int],
    draw_frame_numbers: bool,
    frame_number_scale: float,
) -> None:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None or end_frame < 0:
        end_frame = nframes - 1
    end_frame = min(end_frame, nframes - 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    while idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = draw_projected_skeletons(
            frame,
            projected_by_frame.get(idx),
            edges,
            mouse_colors,
            point_radius,
            line_thickness,
            draw_labels,
        )
        if draw_frame_numbers:
            _draw_frame_number(annotated, idx, font_scale=frame_number_scale)
        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()


def join_videos(top_path: str, front_path: str, side_path: str, out_path: str) -> None:
    cap_t = cv2.VideoCapture(top_path)
    cap_f = cv2.VideoCapture(front_path)
    cap_s = cv2.VideoCapture(side_path)
    if not (cap_t.isOpened() and cap_f.isOpened() and cap_s.isOpened()):
        raise RuntimeError("Could not open one or more videos for joining.")

    fps = cap_t.get(cv2.CAP_PROP_FPS)
    w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create joined video: {out_path}")

    w_left = w // 2
    w_right = w - w_left
    h_top = h // 2
    h_bottom = h - h_top

    while True:
        ok_t, ft = cap_t.read()
        ok_f, ff = cap_f.read()
        ok_s, fs = cap_s.read()
        if not (ok_t and ok_f and ok_s):
            break

        top_small = cv2.resize(ft, (w_right, h_top), interpolation=cv2.INTER_AREA)
        front_small = cv2.resize(ff, (w_right, h_bottom), interpolation=cv2.INTER_AREA)
        side_small = cv2.resize(fs, (w_left, h_bottom), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[0:h_top, w_left:w] = top_small
        canvas[h_top:h, 0:w_left] = side_small
        canvas[h_top:h, w_left:w] = front_small
        writer.write(canvas)

    cap_t.release()
    cap_f.release()
    cap_s.release()
    writer.release()


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overlay_coords3d_on_videos.py",
        description="Overlay skeletons from coords_3d.csv onto top/front/side videos using cameras.json.",
    )
    p.add_argument("--coords-3d", required=True, help="Path to coords_3d.csv with 3D body-part coordinates")
    p.add_argument("--cameras", required=True, help="Path to cameras.json")
    p.add_argument("--template", required=True, help="Skeleton template with [EDGES] section")

    p.add_argument("--topIn", required=True, help="Input top camera video")
    p.add_argument("--frontIn", required=True, help="Input front camera video")
    p.add_argument("--sideIn", required=True, help="Input side camera video")

    p.add_argument("--topOut", required=True, help="Output annotated top video")
    p.add_argument("--frontOut", required=True, help="Output annotated front video")
    p.add_argument("--sideOut", required=True, help="Output annotated side video")
    p.add_argument("--joinedVideo", default=None, help="Optional output path for a joined mosaic video")

    p.add_argument("--topCam", default="cam1_top", help="Camera name for top view")
    p.add_argument("--frontCam", default="cam2_front", help="Camera name for front view")
    p.add_argument("--sideCam", default="cam3_side", help="Camera name for side view")

    p.add_argument("--mouse0-bgr", type=parse_bgr, default=(0, 255, 0), help="BGR color for mouse 0, e.g. 0,255,0")
    p.add_argument("--mouse1-bgr", type=parse_bgr, default=(0, 0, 255), help="BGR color for mouse 1, e.g. 0,0,255")
    p.add_argument("--point-radius", type=int, default=4)
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--draw-labels", action="store_true", help="Draw mouse_id:node labels next to points")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=-1, help="Inclusive end frame; -1 means until video end")
    p.add_argument("--no-frame-numbers", action="store_true", help="Disable frame numbers")
    p.add_argument("--frame-number-scale", type=float, default=1.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cameras = load_cameras_json(args.cameras)
    for name in (args.topCam, args.frontCam, args.sideCam):
        if name not in cameras:
            raise KeyError(f"Camera '{name}' not found in {args.cameras}. Available: {sorted(cameras)}")

    edges = load_template_edges(args.template)
    coords_by_frame = load_coords_3d(args.coords_3d)

    cam_names = [args.topCam, args.frontCam, args.sideCam]
    projected = build_projected_index(coords_by_frame, cameras, cam_names)

    mouse_colors = {
        0: args.mouse0_bgr,
        1: args.mouse1_bgr,
    }

    end_frame = None if args.end_frame is None or int(args.end_frame) < 0 else int(args.end_frame)
    draw_frame_numbers = not bool(args.no_frame_numbers)

    overlay_video(
        args.topIn,
        args.topOut,
        projected_by_frame=projected[args.topCam],
        edges=edges,
        mouse_colors=mouse_colors,
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
        draw_frame_numbers=draw_frame_numbers,
        frame_number_scale=float(args.frame_number_scale),
    )

    overlay_video(
        args.frontIn,
        args.frontOut,
        projected_by_frame=projected[args.frontCam],
        edges=edges,
        mouse_colors=mouse_colors,
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
        draw_frame_numbers=draw_frame_numbers,
        frame_number_scale=float(args.frame_number_scale),
    )

    overlay_video(
        args.sideIn,
        args.sideOut,
        projected_by_frame=projected[args.sideCam],
        edges=edges,
        mouse_colors=mouse_colors,
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
        draw_frame_numbers=draw_frame_numbers,
        frame_number_scale=float(args.frame_number_scale),
    )

    if args.joinedVideo:
        join_videos(args.topOut, args.frontOut, args.sideOut, args.joinedVideo)

    print("Wrote:")
    print(f"  {args.topOut}")
    print(f"  {args.frontOut}")
    print(f"  {args.sideOut}")
    if args.joinedVideo:
        print(f"  {args.joinedVideo}")


if __name__ == "__main__":
    main()
