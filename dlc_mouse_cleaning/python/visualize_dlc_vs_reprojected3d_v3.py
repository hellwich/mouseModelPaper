#!/usr/bin/env python3
"""
visualize_dlc_vs_reprojected3d.py

Overlay two mouse skeleton layers onto camera videos:
  1) Raw DLC 2D detections (drawn with colors at half brightness)
  2) Reprojected 2D points from run_mouse3d 3D output (triangulated3d.csv), drawn in full brightness

This is a stripped-down, visualization-only variant of the provided overlay script, with the same drawing style.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


# -----------------------------
# Camera model (simple pinhole)
# -----------------------------
@dataclass(frozen=True)
class PinholeCamera:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3

    def project(self, Xw: np.ndarray) -> Optional[Tuple[float, float]]:
        """Project a 3D point in world coords into pixel coords. Returns None if behind camera."""
        Xc = self.R @ Xw.reshape(3,) + self.t.reshape(3,)
        z = float(Xc[2])
        if z <= 1e-9:
            return None
        u = self.fx * (float(Xc[0]) / z) + self.cx
        v = self.fy * (float(Xc[1]) / z) + self.cy
        return (u, v)


def load_cameras_json(path: str) -> Dict[str, PinholeCamera]:
    with open(path, "r") as f:
        data = json.load(f)
    cams = {}
    for c in data.get("cameras", []):
        name = c["name"]
        intr = c["intrinsics"]
        ext = c["extrinsics"]
        R = np.array(ext["R"], dtype=np.float64)
        t = np.array(ext["t"], dtype=np.float64)
        cams[name] = PinholeCamera(
            name=name,
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            R=R,
            t=t,
        )
    if not cams:
        raise ValueError(f"No cameras found in {path}")
    return cams


# -----------------------------
# Template / skeleton edges
# -----------------------------
def load_template_edges(template_path: str) -> List[Tuple[str, str]]:
    """
    Parse [EDGES] section from template file.
    Accepts either 'a b' or 'a,b' per line.
    """
    edges: List[Tuple[str, str]] = []
    in_edges = False
    with open(template_path, "r") as f:
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

            # split on comma or whitespace
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))

    if not edges:
        raise ValueError(f"No edges found in [EDGES] section of {template_path}")
    return edges


# -----------------------------
# Colors
# -----------------------------
def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("BGR must be 'B,G,R' (e.g. 0,255,0)")
    b, g, r = (int(float(p)) for p in parts)
    b = max(0, min(255, b))
    g = max(0, min(255, g))
    r = max(0, min(255, r))
    return (b, g, r)

def _draw_frame_number(frame_bgr: np.ndarray, frame_index: int, *, font_scale: float = 1.0) -> None:
    """Overlay the frame index in the top-left corner (OpenCV BGR image).

    Copied in spirit from overlay_dlc_on_videos_v9_clean_frameNumbersVideo_aggstats.py.
    """
    txt = str(int(frame_index))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    margin = int(10 * font_scale)

    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
    x, y = margin, margin + th

    # Background rectangle for readability.
    cv2.rectangle(
        frame_bgr,
        (x - margin, y - th - margin),
        (x + tw + margin, y + baseline + margin),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(frame_bgr, txt, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _hash_color(name: str) -> Tuple[int, int, int]:
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()
    b = int(h[0:2], 16)
    g = int(h[2:4], 16)
    r = int(h[4:6], 16)
    return (b, g, r)


def scale_bgr(bgr: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(c * factor)))) for c in bgr)  # type: ignore


def build_two_mouse_color_maps(
    dlc_individuals: Sequence[str],
    mouse0_bgr: Tuple[int, int, int],
    mouse1_bgr: Tuple[int, int, int],
    *,
    dim_dlc: bool = True,
) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, Tuple[int, int, int]], Dict[str, str]]:
    """
    Returns:
      - colors_dlc: individual->BGR (half brightness if dim_dlc=True; otherwise full brightness)
      - colors_reproj: 'mouse0'/'mouse1'->BGR (full brightness)
      - mapping_info: dlc_individual -> 'mouse0'/'mouse1' (best-effort)
    """
    uniq = list(dict.fromkeys([str(x) for x in dlc_individuals]))
    mapping_info: Dict[str, str] = {}

    # Best-effort mapping if DLC already uses mouse0/mouse1 labels
    dlc_set = set(uniq)
    if "mouse0" in dlc_set or "mouse1" in dlc_set:
        if "mouse0" in dlc_set:
            mapping_info["mouse0"] = "mouse0"
        if "mouse1" in dlc_set:
            mapping_info["mouse1"] = "mouse1"
    else:
        # Otherwise: map first two encountered individuals to mouse0/mouse1.
        if len(uniq) >= 1:
            mapping_info[uniq[0]] = "mouse0"
        if len(uniq) >= 2:
            mapping_info[uniq[1]] = "mouse1"
        # if >2, they will fall back to hashed colors

    colors_reproj = {"mouse0": mouse0_bgr, "mouse1": mouse1_bgr}

    colors_dlc: Dict[str, Tuple[int, int, int]] = {}
    for ind in uniq:
        mapped = mapping_info.get(ind)
        if mapped == "mouse0":
            base = mouse0_bgr
        elif mapped == "mouse1":
            base = mouse1_bgr
        else:
            base = _hash_color(ind)
        colors_dlc[ind] = scale_bgr(base, 0.5) if dim_dlc else base

    return colors_dlc, colors_reproj, mapping_info


# -----------------------------
# Data loading / reprojection
# -----------------------------
def load_dlc_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame", "camera", "individual", "bodypart", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "likelihood" not in df.columns:
        df["likelihood"] = 1.0
    # normalize types
    df["frame"] = df["frame"].astype(int)
    df["camera"] = df["camera"].astype(str)
    df["individual"] = df["individual"].astype(str)
    df["bodypart"] = df["bodypart"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["likelihood"] = df["likelihood"].astype(float)
    return df


def load_triangulated3d(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame", "bodypart", "mouse_id", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["frame"] = df["frame"].astype(int)
    df["bodypart"] = df["bodypart"].astype(str)
    df["mouse_id"] = df["mouse_id"].astype(int)
    for c in ["X", "Y", "Z"]:
        df[c] = df[c].astype(float)
    return df


def build_reprojection_long_df(
    tri_df: pd.DataFrame,
    cameras: Dict[str, PinholeCamera],
    camera_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for r in tri_df.itertuples(index=False):
        Xw = np.array([float(r.X), float(r.Y), float(r.Z)], dtype=np.float64)
        mouse_id = int(r.mouse_id)
        individual = f"mouse{mouse_id}"
        for cam_name in camera_names:
            cam = cameras.get(cam_name)
            if cam is None:
                continue
            uv = cam.project(Xw)
            if uv is None:
                continue
            u, v = uv
            rows.append(
                {
                    "frame": int(r.frame),
                    "camera": cam_name,
                    "individual": individual,
                    "bodypart": str(r.bodypart),
                    "x": float(u),
                    "y": float(v),
                    "likelihood": 1.0,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        # keep schema
        out = pd.DataFrame(columns=["frame", "camera", "individual", "bodypart", "x", "y", "likelihood"])
    return out


def index_by_frame(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    return {int(k): v for k, v in df.groupby("frame", sort=False)}


# -----------------------------
# Drawing (same style as provided overlay script)
# -----------------------------
def overlay_frame_two_layers(
    frame_bgr: np.ndarray,
    rows_dlc: Optional[pd.DataFrame],
    rows_reproj: Optional[pd.DataFrame],
    edges: Optional[List[Tuple[str, str]]],
    colors_dlc: Dict[str, Tuple[int, int, int]],
    colors_reproj: Dict[str, Tuple[int, int, int]],
    likelihood_min: float,
    point_radius: int,
    line_thickness: int,
    draw_bodypart_labels: bool,
) -> np.ndarray:
    """
    Draw DLC first (half-bright colors), then reprojected 3D (full colors).
    """
    out = frame_bgr.copy()

    def _draw_layer(rows: pd.DataFrame, colors: Dict[str, Tuple[int, int, int]]):
        # group by individual
        for individual, group in rows.groupby("individual"):
            col = colors.get(str(individual), (0, 255, 0))
            pts = {}
            for rr in group.itertuples(index=False):
                if float(rr.likelihood) < likelihood_min:
                    continue
                x, y = int(round(float(rr.x))), int(round(float(rr.y)))
                bp = str(rr.bodypart)
                pts[bp] = (x, y)
                cv2.circle(out, (x, y), point_radius, col, -1)
                if draw_bodypart_labels:
                    cv2.putText(
                        out,
                        bp,
                        (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

            # skeleton edges
            if edges is not None:
                for a, b in edges:
                    if a in pts and b in pts:
                        cv2.line(out, pts[a], pts[b], col, line_thickness)

    if rows_dlc is not None and not rows_dlc.empty:
        _draw_layer(rows_dlc, colors_dlc)

    if rows_reproj is not None and not rows_reproj.empty:
        _draw_layer(rows_reproj, colors_reproj)

    return out


def overlay_video(
    in_path: str,
    out_path: str,
    dlc_by_frame: Dict[int, pd.DataFrame],
    reproj_by_frame: Dict[int, pd.DataFrame],
    edges: Optional[List[Tuple[str, str]]],
    colors_dlc: Dict[str, Tuple[int, int, int]],
    colors_reproj: Dict[str, Tuple[int, int, int]],
    likelihood_min: float,
    point_radius: int,
    line_thickness: int,
    draw_bodypart_labels: bool,
    start_frame: int,
    end_frame: Optional[int],
    *,
    draw_dlc: bool = True,
    draw_reproj: bool = True,
    draw_frame_numbers: bool = True,
    frame_number_scale: float = 1.0,
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

    # Seek
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    while idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        rows_dlc = dlc_by_frame.get(idx) if draw_dlc else None
        rows_repr = reproj_by_frame.get(idx) if draw_reproj else None

        annotated = overlay_frame_two_layers(
            frame,
            rows_dlc,
            rows_repr,
            edges=edges,
            colors_dlc=colors_dlc,
            colors_reproj=colors_reproj,
            likelihood_min=likelihood_min,
            point_radius=point_radius,
            line_thickness=line_thickness,
            draw_bodypart_labels=draw_bodypart_labels,
        )

        if draw_frame_numbers:
            _draw_frame_number(annotated, idx, font_scale=frame_number_scale)

        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()



def join_videos(top_path: str, front_path: str, side_path: str, out_path: str) -> None:
    """
    Join videos into a 2x2 mosaic, matching the layout from the provided overlay script:
      [ blank | top ]
      [ side  | front ]
    """
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
    p = argparse.ArgumentParser(prog="visualize_dlc_vs_reprojected3d.py")

    p.add_argument("--dlc", required=True, help="Path to dlc_long.csv")
    p.add_argument("--triangulated3d", required=True, help="Path to tune*_triangulated3d.csv from run_mouse3d output")
    p.add_argument("--cameras", required=True, help="Path to cameras.json (intrinsics+extrinsics)")

    p.add_argument("--topIn", required=True, help="Input top camera video")
    p.add_argument("--frontIn", required=True, help="Input front camera video")
    p.add_argument("--sideIn", required=True, help="Input side camera video")

    p.add_argument("--topOut", required=True, help="Output annotated top video")
    p.add_argument("--frontOut", required=True, help="Output annotated front video")
    p.add_argument("--sideOut", required=True, help="Output annotated side video")

    p.add_argument("--joinedVideo", default=None,
                   help="Optional output path for a joined mosaic video (same resolution as inputs).")

    p.add_argument("--topCam", default="cam1_top", help="Camera name for top view (default cam1_top)")
    p.add_argument("--frontCam", default="cam2_front", help="Camera name for front view (default cam2_front)")
    p.add_argument("--sideCam", default="cam3_side", help="Camera name for side view (default cam3_side)")

    p.add_argument("--template", required=True,
                   help="Skeleton template file with [EDGES] section (e.g., mouse3_mesh_shortTail.txt)")

    p.add_argument("--mouse0-bgr", type=parse_bgr, default="0,255,0",
                   help="B,G,R color for mouse0 in reprojected layer (default 0,255,0)")
    p.add_argument("--mouse1-bgr", type=parse_bgr, default="0,0,255",
                   help="B,G,R color for mouse1 in reprojected layer (default 0,0,255)")

    p.add_argument("--overlay", choices=["both", "dlc", "cleaned"], default="both",
               help="Which layer(s) to overlay: dlc (raw 2D only), cleaned (reprojected 3D only), or both (default).")
    p.add_argument("--no-frame-number", action="store_true",
               help="Disable drawing the frame number on output videos (default: enabled).")
    p.add_argument("--frame-number-scale", type=float, default=1.0,
               help="Font scale for the overlaid frame number (default: 1.0).")

    p.add_argument("--likelihood-min", type=float, default=0.2,
                   help="Only draw DLC points with likelihood >= this (default 0.2). Reprojected points are always drawn.")

    p.add_argument("--draw-labels", action="store_true",
                   help="Draw bodypart labels next to points (default off).")

    p.add_argument("--point-radius", type=int, default=3, help="Point radius in pixels (default 3)")
    p.add_argument("--line-thickness", type=int, default=2, help="Skeleton line thickness (default 2)")

    p.add_argument("--start-frame", type=int, default=0, help="First frame index to process (default 0)")
    p.add_argument("--end-frame", type=int, default=-1,
                   help="Last frame index to process, inclusive (default -1 meaning to video end)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    dlc_df = load_dlc_long(args.dlc)
    tri_df = load_triangulated3d(args.triangulated3d)
    cams = load_cameras_json(args.cameras)

    edges = load_template_edges(args.template)

    # Build reprojection df for the three cameras we actually render
    cam_names = [args.topCam, args.frontCam, args.sideCam]
    reproj_df = build_reprojection_long_df(tri_df, cams, cam_names)
    # Colors / overlay mode
    overlay_mode = str(args.overlay).lower()
    if overlay_mode not in ("both", "dlc", "cleaned"):
        overlay_mode = "both"
    dim_dlc = overlay_mode == "both"
    draw_dlc = overlay_mode in ("both", "dlc")
    draw_reproj = overlay_mode in ("both", "cleaned")
    draw_frame_numbers = not bool(args.no_frame_number)

    colors_dlc, colors_reproj, mapping_info = build_two_mouse_color_maps(
        dlc_df["individual"].unique().tolist(),
        args.mouse0_bgr,
        args.mouse1_bgr,
        dim_dlc=dim_dlc,
    )





    # Inform user about mapping DLC individuals -> mouse0/mouse1 (best-effort)
    if mapping_info:
        print("[INFO] DLC individual -> canonical mouse mapping (best-effort):")
        for k, v in mapping_info.items():
            print(f"  {k} -> {v}")
    else:
        print("[INFO] No DLC individual mapping inferred; using hashed colors for DLC layer.")

    # Split by camera then index by frame (avoids filtering inside the video loop)
    dlc_top = index_by_frame(dlc_df[dlc_df["camera"] == args.topCam])
    dlc_front = index_by_frame(dlc_df[dlc_df["camera"] == args.frontCam])
    dlc_side = index_by_frame(dlc_df[dlc_df["camera"] == args.sideCam])

    rep_top = index_by_frame(reproj_df[reproj_df["camera"] == args.topCam])
    rep_front = index_by_frame(reproj_df[reproj_df["camera"] == args.frontCam])
    rep_side = index_by_frame(reproj_df[reproj_df["camera"] == args.sideCam])

    end_frame = None if args.end_frame is None or int(args.end_frame) < 0 else int(args.end_frame)

    # For DLC layer, we want half-bright colors; for reproj layer we want full brightness.
    # colors_dlc already half-bright; colors_reproj full.
    overlay_video(
        args.topIn, args.topOut,
        dlc_by_frame=dlc_top, reproj_by_frame=rep_top,
        edges=edges, colors_dlc=colors_dlc, colors_reproj=colors_reproj,
        likelihood_min=float(args.likelihood_min),
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_bodypart_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
    draw_dlc=draw_dlc,
    draw_reproj=draw_reproj,
    draw_frame_numbers=draw_frame_numbers,
    frame_number_scale=float(args.frame_number_scale),
    )
    overlay_video(
        args.frontIn, args.frontOut,
        dlc_by_frame=dlc_front, reproj_by_frame=rep_front,
        edges=edges, colors_dlc=colors_dlc, colors_reproj=colors_reproj,
        likelihood_min=float(args.likelihood_min),
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_bodypart_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
    draw_dlc=draw_dlc,
    draw_reproj=draw_reproj,
    draw_frame_numbers=draw_frame_numbers,
    frame_number_scale=float(args.frame_number_scale),
    )
    overlay_video(
        args.sideIn, args.sideOut,
        dlc_by_frame=dlc_side, reproj_by_frame=rep_side,
        edges=edges, colors_dlc=colors_dlc, colors_reproj=colors_reproj,
        likelihood_min=float(args.likelihood_min),
        point_radius=int(args.point_radius),
        line_thickness=int(args.line_thickness),
        draw_bodypart_labels=bool(args.draw_labels),
        start_frame=int(args.start_frame),
        end_frame=end_frame,
    draw_dlc=draw_dlc,
    draw_reproj=draw_reproj,
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
