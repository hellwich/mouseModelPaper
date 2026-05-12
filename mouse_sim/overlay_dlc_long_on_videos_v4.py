
#!/usr/bin/env python3
"""
overlay_dlc_long_on_videos.py

Overlay DeepLabCut 2D predictions (from a dlc_long.csv produced by dlc_read_multicam_cli*.py)
onto per-camera videos, draw a skeleton (edges from a template .txt), and optionally produce
a joined/mosaic video.

This intentionally does NOT use cameras.json or any 3D->2D reprojection; it reads 2D points
directly from dlc_long.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Install with: pip install pandas") from e

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires opencv-python. Install with: pip install opencv-python") from e


# --------------------------
# Helpers
# --------------------------



def _default_palette() -> List[Tuple[int, int, int]]:
    """Return a small BGR palette (OpenCV uses BGR).

    The first colors are chosen to be high-contrast on natural backgrounds.
    """
    return [
        (0, 255, 255),   # yellow
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (0, 0, 255),     # red
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (255, 255, 255), # white
        (0, 165, 255),   # orange
        (203, 192, 255), # pinkish
        (128, 128, 0),   # olive
    ]
def _norm_label(s: str) -> str:
    """Normalize names to match across template and dlc_long (lowercase, underscores)."""
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def _parse_bgr(spec: str) -> Tuple[int, int, int]:
    """
    Parse BGR color from:
      - "B,G,R" integers 0..255
      - "#RRGGBB" (will be converted to BGR)
    """
    spec = spec.strip()
    if spec.startswith("#") and len(spec) == 7:
        r = int(spec[1:3], 16)
        g = int(spec[3:5], 16)
        b = int(spec[5:7], 16)
        return (b, g, r)
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid color '{spec}'. Use 'B,G,R' or '#RRGGBB'.")
    b, g, r = (int(parts[0]), int(parts[1]), int(parts[2]))
    for v in (b, g, r):
        if v < 0 or v > 255:
            raise ValueError(f"Invalid color '{spec}': values must be 0..255.")
    return (b, g, r)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



def _draw_frame_number(frame: np.ndarray, frame_idx: int) -> None:
    """Draw current frame index on the image (top-left)."""
    txt = f"frame {frame_idx:06d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(txt, font, scale, thickness)
    x, y = 10, 10 + th
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame, txt, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# --------------------------
# Template skeleton edges
# --------------------------

def load_template_edges(template_txt: Path) -> List[Tuple[str, str]]:
    """
    Parse edges from a mouse graph / template file.

    We support a few common formats:
    - Lines like: "A B" (two tokens) under an [EDGES] block
    - CSV-like: "A,B"
    - Lines like: "A->B" or "A - B"
    """
    edges: List[Tuple[str, str]] = []
    in_edges = False
    text = template_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # section headers
        if re.match(r"^\[.*\]$", line):
            in_edges = line.strip("[]").strip().lower() in {"edges", "skeleton", "graph"}
            continue

        # If the file has explicit sections, only parse inside [EDGES]/[SKELETON]/[GRAPH]
        if any(re.match(r"^\[.*\]$", l.strip()) for l in text):
            if not in_edges:
                continue

        # Parse edge line
        a = b = None
        if "->" in line:
            parts = [p.strip() for p in line.split("->", 1)]
            if len(parts) == 2:
                a, b = parts
        elif "," in line and not line.startswith("http"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
        elif "-" in line and re.search(r"\s-\s", line):
            parts = [p.strip() for p in line.split("-", 1)]
            if len(parts) == 2:
                a, b = parts
        else:
            toks = line.split()
            if len(toks) >= 2:
                a, b = toks[0], toks[1]

        if a and b:
            edges.append((_norm_label(a), _norm_label(b)))

    # De-duplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for e in edges:
        if e not in seen and (e[1], e[0]) not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq


# --------------------------
# DLC long format loader
# --------------------------

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    likelihood: float


def load_dlc_long_csv(path: Path) -> pd.DataFrame:
    """
    Expected columns (case-insensitive):
      frame, camera, individual, bodypart, x, y, likelihood

    Extra columns are ignored.
    """
    df = pd.read_csv(path)
    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"frame", "camera", "individual", "bodypart", "x", "y", "likelihood"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    # type normalize
    df["frame"] = df["frame"].astype(int)
    df["camera"] = df["camera"].astype(str)
    df["individual"] = df["individual"].astype(str)
    df["bodypart"] = df["bodypart"].astype(str)
    for c in ("x", "y", "likelihood"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize labels for matching to template
    df["bodypart_norm"] = df["bodypart"].map(_norm_label)
    df["individual_norm"] = df["individual"].map(_norm_label)
    df["camera_norm"] = df["camera"].map(_norm_label)
    return df


def build_index(
    df_long: pd.DataFrame,
    pcutoff: Optional[float],
    cameras_keep: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, Dict[str, Dict[str, Point2D]]]]:
    """
    Returns nested dict:
      cam_norm -> frame -> individual_norm -> bodypart_norm -> Point2D

    If pcutoff is not None, we still store points but we keep likelihood so the
    renderer can decide (blob vs circle). We only drop points with NaN x/y.
    """
    if cameras_keep is not None:
        keep = {_norm_label(c) for c in cameras_keep}
        df_long = df_long[df_long["camera_norm"].isin(keep)].copy()

    # drop rows with no coordinates
    df_long = df_long[np.isfinite(df_long["x"]) & np.isfinite(df_long["y"])].copy()

    idx: Dict[str, Dict[int, Dict[str, Dict[str, Point2D]]]] = {}
    # iterate rows (fast enough for typical sizes)
    for row in df_long.itertuples(index=False):
        cam = getattr(row, "camera_norm")
        frame = int(getattr(row, "frame"))
        ind = getattr(row, "individual_norm")
        bp = getattr(row, "bodypart_norm")
        x = float(getattr(row, "x"))
        y = float(getattr(row, "y"))
        lik = float(getattr(row, "likelihood"))
        idx.setdefault(cam, {}).setdefault(frame, {}).setdefault(ind, {})[bp] = Point2D(x=x, y=y, likelihood=lik)
    return idx


# --------------------------
# Video overlay
# --------------------------

def _draw_circle(
    img: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int,
    filled: bool,
    thickness: int = 2,
) -> None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    center = (int(round(x)), int(round(y)))
    if filled:
        cv2.circle(img, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        cv2.circle(img, center, radius, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_line(
    img: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return
    cv2.line(
        img,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def overlay_video_with_dlc(
    video_in: Path,
    video_out: Path,
    cam_name: str,
    dlc_index: Dict[str, Dict[int, Dict[str, Dict[str, Point2D]]]],
    edges: List[Tuple[str, str]],
    pcutoff: float,
    colors_by_individual: Dict[str, Tuple[int, int, int]],
    draw_labels: bool = False,
    label_scale: float = 0.4,
    radius: int = 4,
    thickness: int = 2,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[int, float, Tuple[int, int]]:
    """
    Returns (n_frames_written, fps, (w,h)).
    """
    cam_norm = _norm_label(cam_name)
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1

    if end_frame is None:
        end_frame = total_frames - 1 if total_frames > 0 else None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_dir(video_out.parent)
    out = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer: {video_out}")

    # seek to start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    written = 0
    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break

        pts_by_ind = dlc_index.get(cam_norm, {}).get(frame_idx, {})

        # Draw skeleton first (so points are on top)
        for ind_norm, pts in pts_by_ind.items():
            col = colors_by_individual.get(ind_norm, (0, 255, 255))
            for a, b in edges:
                pa = pts.get(a)
                pb = pts.get(b)
                if pa is None or pb is None:
                    continue
                # only draw if both are reasonably confident
                if pa.likelihood >= pcutoff and pb.likelihood >= pcutoff:
                    _draw_line(frame, (pa.x, pa.y), (pb.x, pb.y), col, thickness=thickness)

        # Draw points
        for ind_norm, pts in pts_by_ind.items():
            col = colors_by_individual.get(ind_norm, (0, 255, 255))
            for bp, p in pts.items():
                hi = p.likelihood >= pcutoff
                _draw_circle(frame, p.x, p.y, col, radius=radius, filled=hi, thickness=thickness)
                if draw_labels:
                    text = f"{ind_norm}:{bp}"
                    cv2.putText(
                        frame,
                        text,
                        (int(round(p.x)) + 5, int(round(p.y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_scale,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

        _draw_frame_number(frame, frame_idx)
        out.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    out.release()
    return written, fps, (w, h)


def join_three_videos_quadrants(
    top_video: Path,
    front_video: Path,
    side_video: Path,
    out_video: Path,
    *,
    labels: Optional[Tuple[str, str, str]] = ("top", "front", "side"),
) -> None:
    """Compose a 2x2 mosaic:
    - top camera:   upper-right quadrant
    - front camera: lower-right quadrant
    - side camera:  lower-left quadrant
    - upper-left quadrant left black
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cap_top = cv2.VideoCapture(str(top_video))
    cap_front = cv2.VideoCapture(str(front_video))
    cap_side = cv2.VideoCapture(str(side_video))
    if not cap_top.isOpened():
        raise RuntimeError(f"Could not open video: {top_video}")
    if not cap_front.isOpened():
        raise RuntimeError(f"Could not open video: {front_video}")
    if not cap_side.isOpened():
        raise RuntimeError(f"Could not open video: {side_video}")

    fps = cap_top.get(cv2.CAP_PROP_FPS) or cap_front.get(cv2.CAP_PROP_FPS) or cap_side.get(cv2.CAP_PROP_FPS) or 30.0

    w_top = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_top = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_front = int(cap_front.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_front = int(cap_front.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_side = int(cap_side.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_side = int(cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = max(w_top, w_front, w_side)
    out_h = max(h_top, h_front, h_side)

    # ensure even so we can split exactly into quadrants
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1

    q_w, q_h = out_w // 2, out_h // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_video}")

    def _draw_quadrant_label(canvas: np.ndarray, text: str, origin_xy: Tuple[int, int]) -> None:
        """Draw a label in the top-left corner of a quadrant."""
        if not text:
            return
        x0, y0 = origin_xy
        cv2.putText(
            canvas,
            text,
            (x0 + 6, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    try:
        while True:
            ok_t, fr_t = cap_top.read()
            ok_f, fr_f = cap_front.read()
            ok_s, fr_s = cap_side.read()
            if not (ok_t and ok_f and ok_s):
                break

            fr_t = cv2.resize(fr_t, (q_w, q_h), interpolation=cv2.INTER_AREA)
            fr_f = cv2.resize(fr_f, (q_w, q_h), interpolation=cv2.INTER_AREA)
            fr_s = cv2.resize(fr_s, (q_w, q_h), interpolation=cv2.INTER_AREA)

            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            # upper-left is blank
            canvas[0:q_h, q_w:out_w] = fr_t              # upper-right
            canvas[q_h:out_h, 0:q_w] = fr_s              # lower-left
            canvas[q_h:out_h, q_w:out_w] = fr_f          # lower-right

            # Optional orientation labels.
            if labels is not None:
                try:
                    l_top, l_front, l_side = labels
                except Exception:
                    l_top, l_front, l_side = "top", "front", "side"
                _draw_quadrant_label(canvas, l_top, (q_w, 0))
                _draw_quadrant_label(canvas, l_front, (q_w, q_h))
                _draw_quadrant_label(canvas, l_side, (0, q_h))

            writer.write(canvas)
    finally:
        cap_top.release()
        cap_front.release()
        cap_side.release()
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay DLC 2D points (dlc_long.csv) onto videos and produce a joined mosaic video."
    )
    ap.add_argument("--dlc-long", required=True, type=Path, help="Path to dlc_long.csv")
    ap.add_argument("--template", required=True, type=Path, help="Template .txt with skeleton edges")
    ap.add_argument("--pcutoff", type=float, default=0.6, help="Likelihood threshold for 'confident' points/edges")
    ap.add_argument("--draw-labels", action="store_true", help="Draw text labels (individual:bodypart) near points")
    ap.add_argument("--radius", type=int, default=4, help="Point radius in pixels")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness in pixels")

    ap.add_argument("--top-in", required=True, type=Path, help="Top camera input video")
    ap.add_argument("--front-in", required=True, type=Path, help="Front camera input video")
    ap.add_argument("--side-in", required=True, type=Path, help="Side camera input video")

    ap.add_argument("--top-cam", default="cam1_top", help="Camera name as it appears in dlc_long.csv for top video")
    ap.add_argument("--front-cam", default="cam2_front", help="Camera name as it appears in dlc_long.csv for front video")
    ap.add_argument("--side-cam", default="cam3_side", help="Camera name as it appears in dlc_long.csv for side video")

    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for annotated videos")
    ap.add_argument("--top-out", type=Path, default=None, help="Output path for annotated top video (optional)")
    ap.add_argument("--front-out", type=Path, default=None, help="Output path for annotated front video (optional)")
    ap.add_argument("--side-out", type=Path, default=None, help="Output path for annotated side video (optional)")
    ap.add_argument("--joined-out", type=Path, default=None, help="Output path for joined mosaic video (optional)")

    ap.add_argument("--start-frame", type=int, default=0, help="Start frame index (0-based)")
    ap.add_argument("--end-frame", type=int, default=None, help="End frame index (inclusive); omit for full length")

    ap.add_argument(
        "--ind-colors",
        action="append",
        default=[],
        help="Optional per-individual colors: 'mouse0:#FF00FF' or 'mouse1:0,255,255' (repeatable). "
             "Names are matched after normalization.",
    )

    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    # default output paths
    top_out = args.top_out or (args.out_dir / f"{Path(args.top_in).stem}_dlc_overlay.mp4")
    front_out = args.front_out or (args.out_dir / f"{Path(args.front_in).stem}_dlc_overlay.mp4")
    side_out = args.side_out or (args.out_dir / f"{Path(args.side_in).stem}_dlc_overlay.mp4")
    joined_out = args.joined_out or (args.out_dir / "joined_dlc_overlay.mp4")

    edges = load_template_edges(args.template)
    if not edges:
        print(f"[warn] No edges parsed from template: {args.template}. Will draw points only.")

    df_long = load_dlc_long_csv(args.dlc_long)

    # Build individual color map
    palette = _default_palette()
    colors_by_ind: Dict[str, Tuple[int, int, int]] = {}
    # from CLI
    for spec in args.ind_colors:
        if ":" not in spec:
            raise SystemExit(f"--ind-colors expects 'name:color', got: {spec}")
        name, colspec = spec.split(":", 1)
        colors_by_ind[_norm_label(name)] = _parse_bgr(colspec)

    # fill with defaults for remaining individuals in file
    inds = sorted(set(df_long["individual_norm"].unique()))
    for i, ind in enumerate(inds):
        if ind not in colors_by_ind:
            colors_by_ind[ind] = palette[i % len(palette)]

    # Index just these three cameras (by their dlc_long name)
    dlc_index = build_index(df_long, pcutoff=args.pcutoff, cameras_keep=[args.top_cam, args.front_cam, args.side_cam])

    print(f"[info] dlc_long: {args.dlc_long}")
    print(f"[info] cameras: top={args.top_cam} front={args.front_cam} side={args.side_cam}")
    print(f"[info] individuals: {inds}")
    print(f"[info] edges: {len(edges)}")
    print(f"[info] writing to: {args.out_dir}")

    overlay_video_with_dlc(
        args.top_in, top_out, args.top_cam, dlc_index, edges, args.pcutoff, colors_by_ind,
        draw_labels=args.draw_labels, radius=args.radius, thickness=args.thickness,
        start_frame=args.start_frame, end_frame=args.end_frame,
    )
    overlay_video_with_dlc(
        args.front_in, front_out, args.front_cam, dlc_index, edges, args.pcutoff, colors_by_ind,
        draw_labels=args.draw_labels, radius=args.radius, thickness=args.thickness,
        start_frame=args.start_frame, end_frame=args.end_frame,
    )
    overlay_video_with_dlc(
        args.side_in, side_out, args.side_cam, dlc_index, edges, args.pcutoff, colors_by_ind,
        draw_labels=args.draw_labels, radius=args.radius, thickness=args.thickness,
        start_frame=args.start_frame, end_frame=args.end_frame,
    )

    join_three_videos_quadrants(top_out, front_out, side_out, joined_out, labels=("top", "front", "side"))
    print(f"Done.\n  top:    {top_out}\n  front:  {front_out}\n  side:   {side_out}\n  joined: {joined_out}")


if __name__ == "__main__":
    main()