#!/usr/bin/env python3
"""
mice_seg_v3.py

Changes vs v2 (requested):
- Oversampling can use your frame-wise ethogram CSV "as is" (one row per frame with columns like
  interaction, mouse0_rear, mouse1_rear, etc.). Oversampling boosts frames where ANY chosen signal column == 1.

- Manifest can optionally contain an `ethogram_path` column (one ethogram CSV per sample/video).
  If `ethogram_path` is missing but `sample_id` looks like a .csv path that exists, it is treated as the ethogram path.
  If `sample_id` column is missing, it is set to `ethogram_path` (or an auto-generated id).

- Mask videos can be either:
  (a) indexed labels 0/1/2 (background/mouse0/mouse1) stored as grayscale OR 3-channel with identical channels, OR
  (b) color-coded BGR masks (bg black, mouse0/mouse1 colors).
  Auto-detection chooses indexed mode when channels are identical and values are within {0,1,2}.

Everything else stays the same: RGB_t + prev_mask_{t-1} + view_onehot -> 3-class segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as AmpGradScaler
    _HAS_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast as amp_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as AmpGradScaler  # type: ignore
    _HAS_TORCH_AMP = False


def autocast_ctx(enabled: bool, device: torch.device):
    if not enabled:
        return contextlib.nullcontext()
    if device.type != "cuda":
        return contextlib.nullcontext()
    # torch.amp.autocast needs a device_type argument; torch.cuda.amp.autocast does not.
    if _HAS_TORCH_AMP:
        return amp_autocast(device_type="cuda", enabled=True)
    return amp_autocast(enabled=True)


def make_scaler(enabled: bool, device: torch.device):
    if device.type != "cuda":
        return AmpGradScaler(enabled=False)
    if _HAS_TORCH_AMP:
        return AmpGradScaler("cuda", enabled=enabled)
    return AmpGradScaler(enabled=enabled)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import cv2
except ImportError as e:
    raise SystemExit("Please install opencv (conda-forge opencv recommended).") from e


# ----------------------------
# Constants / helpers
# ----------------------------

VIEWS = {"top": 0, "front": 1, "side": 2}
IDX_TO_VIEW = {v: k for k, v in VIEWS.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_frames(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected B,G,R like '25,35,40' but got: {s}")
    try:
        b, g, r = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Non-integer in BGR string: {s}")
    for v in (b, g, r):
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError(f"BGR values must be 0..255 but got: {s}")
    return (b, g, r)


def fourcc_for_path(path: Path) -> int:
    ext = path.suffix.lower()
    if ext == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def resolve_path(base: Path, p: str) -> Path:
    p = p.strip()
    if not p:
        return Path("")
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (base / pp)


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------
# GT mask decoding (indexed OR color-coded BGR)
# ----------------------------

@dataclass(frozen=True)
class MaskColorSpec:
    mouse0_bgr: Tuple[int, int, int] = (25, 35, 40)
    mouse1_bgr: Tuple[int, int, int] = (40, 35, 25)
    bg_bgr: Tuple[int, int, int] = (0, 0, 0)
    tol: int = 0  # per-channel absolute tolerance


def _match_color(mask_bgr: np.ndarray, bgr: Tuple[int, int, int], tol: int) -> np.ndarray:
    b, g, r = bgr
    if tol <= 0:
        return (
            (mask_bgr[:, :, 0] == b)
            & (mask_bgr[:, :, 1] == g)
            & (mask_bgr[:, :, 2] == r)
        )
    mb = mask_bgr[:, :, 0].astype(np.int16)
    mg = mask_bgr[:, :, 1].astype(np.int16)
    mr = mask_bgr[:, :, 2].astype(np.int16)
    return (
        (np.abs(mb - b) <= tol)
        & (np.abs(mg - g) <= tol)
        & (np.abs(mr - r) <= tol)
    )


def _try_decode_indexed_from_3ch(mask_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    If the 3-channel image looks like an indexed label mask:
      - all three channels are identical
      - values are within {0,1,2}
    return HxW uint8 labels, else None.
    """
    c0 = mask_bgr[:, :, 0]
    if not (np.array_equal(c0, mask_bgr[:, :, 1]) and np.array_equal(c0, mask_bgr[:, :, 2])):
        return None
    u = np.unique(c0)
    if u.size == 0:
        return c0.astype(np.uint8)
    if u.min() < 0 or u.max() > 2:
        return None
    return c0.astype(np.uint8)


def decode_mask_to_labels(mask_arr: np.ndarray, color_spec: MaskColorSpec) -> np.ndarray:
    """
    Returns HxW uint8 labels in {0:bg, 1:mouse0, 2:mouse1}.
    Supports:
      - indexed 2D masks with values 0/1/2
      - indexed 3-channel masks (channels identical, values 0/1/2)
      - color-coded 3-channel masks in BGR according to color_spec
    """
    if mask_arr.ndim == 2:
        raw = mask_arr.astype(np.int64)
        u = np.unique(raw)
        if u.size and (u.min() < 0 or u.max() > 2):
            raise ValueError(f"Indexed mask has values outside 0..2: {u[:10].tolist()}")
        return raw.astype(np.uint8)

    if mask_arr.ndim == 3 and mask_arr.shape[2] in (3, 4):
        if mask_arr.shape[2] == 4:
            mask_arr = mask_arr[:, :, :3]
        mask_bgr = mask_arr  # cv2 gives BGR

        # First try: indexed labels stored in 3 channels
        maybe = _try_decode_indexed_from_3ch(mask_bgr)
        if maybe is not None:
            return maybe

        # Otherwise: treat as color-coded
        bg = _match_color(mask_bgr, color_spec.bg_bgr, color_spec.tol)
        m0 = _match_color(mask_bgr, color_spec.mouse0_bgr, color_spec.tol)
        m1 = _match_color(mask_bgr, color_spec.mouse1_bgr, color_spec.tol)

        out = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
        out[m0] = 1
        out[m1] = 2

        unknown = ~(bg | m0 | m1)
        if np.any(unknown):
            bad_pixels = mask_bgr[unknown]
            uniq = np.unique(bad_pixels.reshape(-1, 3), axis=0)
            uniq = uniq[:10]
            raise ValueError(
                "GT mask contains colors not matching either indexed 0/1/2 OR "
                f"bg={color_spec.bg_bgr}, mouse0={color_spec.mouse0_bgr}, mouse1={color_spec.mouse1_bgr}. "
                f"First unknown BGR colors: {uniq.tolist()} (tol={color_spec.tol})"
            )
        return out

    raise ValueError(f"Unsupported mask shape: {mask_arr.shape}")


def read_mask_labels(path: Path, color_spec: MaskColorSpec) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return decode_mask_to_labels(m, color_spec)


# ----------------------------
# Camera.json handling (assert + cage ROI crop)
# ----------------------------

_CAMCACHE: Dict[str, dict] = {}


def load_cameras_json(path: Path) -> dict:
    key = str(path.resolve())
    if key in _CAMCACHE:
        return _CAMCACHE[key]
    with path.open("r") as f:
        data = json.load(f)
    if "cameras" not in data or not isinstance(data["cameras"], list):
        raise ValueError(f"{path} does not look like a cameras.json (missing 'cameras' list)")
    if "cage" not in data:
        raise ValueError(f"{path} does not look like a cameras.json (missing 'cage')")
    _CAMCACHE[key] = data
    return data


def camera_resolution_for_view(cameras_data: dict, view: str) -> Tuple[int, int, str]:
    view = view.lower().strip()
    cams = cameras_data["cameras"]
    for cam in cams:
        name = str(cam.get("name", "")).lower()
        if name.endswith(f"_{view}"):
            return int(cam["width"]), int(cam["height"]), cam.get("name", "")
    for cam in cams:
        name = str(cam.get("name", "")).lower()
        if f"_{view}" in name:
            return int(cam["width"]), int(cam["height"]), cam.get("name", "")
    raise KeyError(f"No camera found in cameras.json matching view='{view}'")


def source_resolution_rgb(path: Path) -> Tuple[int, int]:
    if path.is_dir():
        frames = list_frames(path)
        if not frames:
            raise FileNotFoundError(f"No frames in folder: {path}")
        img = read_image_rgb(frames[0])
        h, w = img.shape[:2]
        return (w, h)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame from: {path}")
    h, w = frame_bgr.shape[:2]
    return (w, h)


def assert_matches_camera(view: str, rgb_path: Path, cameras_json_path: Path) -> None:
    cams = load_cameras_json(cameras_json_path)
    exp_w, exp_h, cam_name = camera_resolution_for_view(cams, view)
    act_w, act_h = source_resolution_rgb(rgb_path)
    if (act_w, act_h) != (exp_w, exp_h):
        raise ValueError(
            f"Resolution mismatch for view='{view}' camera='{cam_name}': "
            f"expected {exp_w}x{exp_h} (from {cameras_json_path}), got {act_w}x{act_h} (from {rgb_path})."
        )


@dataclass(frozen=True)
class CameraModel:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray


def camera_for_view(cameras_data: dict, view: str) -> CameraModel:
    view = view.lower().strip()
    cams = cameras_data["cameras"]

    def to_cam(cam: dict) -> CameraModel:
        return CameraModel(
            name=str(cam["name"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            R=np.array(cam["R"], dtype=np.float64),
            t=np.array(cam["t"], dtype=np.float64),
        )

    for cam in cams:
        if str(cam.get("name", "")).lower().endswith(f"_{view}"):
            return to_cam(cam)
    for cam in cams:
        if f"_{view}" in str(cam.get("name", "")).lower():
            return to_cam(cam)
    raise KeyError(f"No camera found matching view='{view}' in cameras.json")


def cage_corners_world(cameras_data: dict) -> np.ndarray:
    cage = cameras_data["cage"]
    ox, oy, oz = (float(cage["origin"][0]), float(cage["origin"][1]), float(cage["origin"][2]))
    w = float(cage["width"])
    d = float(cage["depth"])
    h = float(cage["height"])
    corners = []
    for dx in (0.0, w):
        for dy in (0.0, d):
            for dz in (0.0, h):
                corners.append([ox + dx, oy + dy, oz + dz])
    return np.array(corners, dtype=np.float64)


def project_world_to_image(cam: CameraModel, Pw: np.ndarray) -> np.ndarray:
    Pc = (cam.R @ Pw.T).T + cam.t.reshape(1, 3)
    z = Pc[:, 2]
    uv = np.full((Pw.shape[0], 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    uv[valid, 0] = cam.fx * (Pc[valid, 0] / z[valid]) + cam.cx
    uv[valid, 1] = cam.fy * (Pc[valid, 1] / z[valid]) + cam.cy
    return uv


def cage_crop_box(cameras_json_path: Path, view: str, margin: int = 10) -> Tuple[int, int, int, int]:
    cams_data = load_cameras_json(cameras_json_path)
    cam = camera_for_view(cams_data, view)
    Pw = cage_corners_world(cams_data)
    uv = project_world_to_image(cam, Pw)

    good = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    if int(good.sum()) < 4:
        raise ValueError(f"Too few valid cage corner projections for view='{view}' (got {int(good.sum())}).")

    xs = uv[good, 0]
    ys = uv[good, 1]
    x0 = int(math.floor(xs.min())) - margin
    y0 = int(math.floor(ys.min())) - margin
    x1 = int(math.ceil(xs.max())) + margin
    y1 = int(math.ceil(ys.max())) + margin

    x0 = max(0, min(x0, cam.width - 1))
    y0 = max(0, min(y0, cam.height - 1))
    x1 = max(x0 + 1, min(x1, cam.width))
    y1 = max(y0 + 1, min(y1, cam.height))
    return (x0, y0, x1, y1)

def letterbox_params(in_w: int, in_h: int, out_w: int, out_h: int):
    """
    Computes letterbox resize geometry consistently.
    Returns: (nw, nh, pad_x, pad_y)
      - resized content is (nw, nh)
      - placed at (pad_x, pad_y) in the out_w/out_h canvas
    """
    scale = min(out_w / in_w, out_h / in_h)
    nw = int(round(in_w * scale))
    nh = int(round(in_h * scale))

    # Clamp in case rounding pushes by 1
    nw = max(1, min(nw, out_w))
    nh = max(1, min(nh, out_h))

    pad_x = (out_w - nw) // 2
    pad_y = (out_h - nh) // 2
    return nw, nh, pad_x, pad_y

# ----------------------------
# Resizing / augmentation
# ----------------------------

def resize_img(img: np.ndarray, out_hw: Tuple[int, int], mode: str) -> np.ndarray:
    out_h, out_w = out_hw
    if mode == "stretch":
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    nw, nh, pad_x, pad_y = letterbox_params(w, h, out_w, out_h)

    img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_s
    return canvas

def resize_mask(mask: np.ndarray, out_hw: Tuple[int, int], mode: str) -> np.ndarray:
    out_h, out_w = out_hw
    if mode == "stretch":
        return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    nw, nh, pad_x, pad_y = letterbox_params(w, h, out_w, out_h)

    m_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((out_h, out_w), dtype=mask.dtype)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = m_s
    return canvas

def apply_augment(img: np.ndarray, mask_t: np.ndarray, mask_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()
        mask_t = mask_t[:, ::-1].copy()
        mask_prev = mask_prev[:, ::-1].copy()
    if random.random() < 0.3:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    return img, mask_t, mask_prev


def to_one_hot(mask_hw: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    return F.one_hot(mask_hw.long(), num_classes=num_classes).permute(2, 0, 1).float()


def view_one_hot(view_idx: int, h: int, w: int) -> torch.Tensor:
    v = torch.zeros(3, h, w, dtype=torch.float32)
    v[view_idx, :, :] = 1.0
    return v


# ----------------------------
# Video IO
# ----------------------------

class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self._len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self) -> int:
        return self._len

    def fps(self) -> float:
        v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return v if v > 1e-3 else 30.0

    def read_bgr(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed reading frame {idx} from {self.path}")
        return frame_bgr

    def read_rgb(self, idx: int) -> np.ndarray:
        return cv2.cvtColor(self.read_bgr(idx), cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ----------------------------
# Manifest / Ethogram (frame-wise)
# ----------------------------

@dataclass
class SampleInfo:
    sample_id: str
    view: str
    rgb_path: Path
    mask_path: Path
    camera_path: Optional[Path] = None
    ethogram_path: Optional[Path] = None


def load_manifest(path: Path) -> List[SampleInfo]:
    base = path.parent
    items: List[SampleInfo] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"view", "rgb_path", "mask_path"}
        missing = required - fields
        if missing:
            raise ValueError(f"manifest.csv missing columns: {missing}")

        has_sample_id = "sample_id" in fields
        has_ethogram_path = "ethogram_path" in fields

        for i, row in enumerate(reader):
            view = row["view"].strip().lower()
            if view not in VIEWS:
                raise ValueError(f"Unknown view '{view}' in manifest row {i}")

            rgb_path = resolve_path(base, row["rgb_path"])
            mask_path = resolve_path(base, row["mask_path"])
            cam = resolve_path(base, row.get("camera_path", "")) if "camera_path" in fields else Path("")
            eth = resolve_path(base, row.get("ethogram_path", "")) if has_ethogram_path else Path("")

            sample_id = row.get("sample_id", "").strip() if has_sample_id else ""
            if not sample_id:
                if eth and eth.suffix.lower() == ".csv":
                    sample_id = str(eth)
                else:
                    sample_id = f"sample_{i:04d}"

            # convenience: if ethogram_path column missing/empty but sample_id is a CSV path, use it
            ethogram_path = eth if eth and str(eth) else None
            sid_as_path = Path(sample_id)
            if ethogram_path is None and sid_as_path.suffix.lower() == ".csv" and sid_as_path.exists():
                ethogram_path = sid_as_path

            items.append(
                SampleInfo(
                    sample_id=sample_id,
                    view=view,
                    rgb_path=rgb_path,
                    mask_path=mask_path,
                    camera_path=cam if str(cam) else None,
                    ethogram_path=ethogram_path,
                )
            )
    return items


def load_framewise_ethogram_flags(
    ethogram_csv: Path,
    num_frames: int,
    oversample_signals: List[str],
) -> np.ndarray:
    """
    Returns a boolean array flags[t] where True means oversample that frame.
    Supports ethograms with at least a 'frame' column OR assumes row order is frame index.
    Signal columns are treated as 1/0 (strings ok).
    """
    flags = np.zeros(num_frames, dtype=np.bool_)
    if not oversample_signals:
        return flags

    with ethogram_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not cols:
            return flags

        has_frame = "frame" in cols
        for row_i, row in enumerate(reader):
            t = row_i
            if has_frame:
                try:
                    t = int(float(row["frame"]))
                except Exception:
                    continue
            if t < 0 or t >= num_frames:
                continue

            hit = False
            for sig in oversample_signals:
                if sig not in row:
                    continue
                v = row[sig]
                try:
                    hit = (int(float(v)) == 1)
                except Exception:
                    hit = (str(v).strip().lower() in {"1", "true", "yes", "y"})
                if hit:
                    break
            flags[t] = hit

    return flags


# ----------------------------
# Dataset
# ----------------------------

class MicePairDataset(Dataset):
    """
    x: (9,H,W) = RGB(3) + prev_mask_onehot(3) + view_onehot(3)
    y: (H,W) labels {0,1,2}
    """

    def __init__(
        self,
        samples: List[SampleInfo],
        out_hw: Tuple[int, int],
        resize_mode: str,
        augment: bool,
        gt_color_spec: MaskColorSpec,
        oversample_signals: Optional[List[str]] = None,
        oversample_factor: float = 3.0,
        min_t: int = 1,
        camera_crop: bool = False,
        camera_crop_margin: int = 10,
    ):
        self.samples = samples
        self.out_hw = out_hw
        self.resize_mode = resize_mode
        self.augment = augment
        self.gt_color_spec = gt_color_spec

        self.oversample_signals = [s.strip() for s in (oversample_signals or []) if s.strip()]
        self.oversample_factor = float(oversample_factor)

        self.camera_crop = bool(camera_crop)
        self.camera_crop_margin = int(camera_crop_margin)
        self._crop_cache: Dict[int, Tuple[int, int, int, int]] = {}
        if self.camera_crop:
            for si, info in enumerate(samples):
                if info.camera_path is not None:
                    self._crop_cache[si] = cage_crop_box(info.camera_path, info.view, margin=self.camera_crop_margin)

        self.index: List[Tuple[int, int]] = []
        self.weights: List[float] = []

        for si, info in enumerate(samples):
            n = self._num_frames_rgb(info)
            flags = None
            if self.oversample_signals and info.ethogram_path is not None:
                flags = load_framewise_ethogram_flags(info.ethogram_path, n, self.oversample_signals)

            for t in range(min_t, n):
                self.index.append((si, t))
                w = 1.0
                if flags is not None and bool(flags[t]):
                    w = self.oversample_factor
                self.weights.append(w)

    def _num_frames_rgb(self, info: SampleInfo) -> int:
        if info.rgb_path.is_dir():
            return len(list_frames(info.rgb_path))
        vr = VideoReader(info.rgb_path)
        n = len(vr)
        vr.close()
        return n

    def _read_rgb(self, info: SampleInfo, t: int) -> np.ndarray:
        if info.rgb_path.is_dir():
            frames = list_frames(info.rgb_path)
            return read_image_rgb(frames[t])
        vr = VideoReader(info.rgb_path)
        img = vr.read_rgb(t)
        vr.close()
        return img

    def _read_mask_labels(self, info: SampleInfo, t: int) -> np.ndarray:
        if info.mask_path.is_dir():
            frames = list_frames(info.mask_path)
            return read_mask_labels(frames[t], self.gt_color_spec)
        vr = VideoReader(info.mask_path)
        frame_bgr = vr.read_bgr(t)
        vr.close()
        return decode_mask_to_labels(frame_bgr, self.gt_color_spec)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        si, t = self.index[idx]
        info = self.samples[si]
        view_idx = VIEWS[info.view]

        img = self._read_rgb(info, t)
        mask_t = self._read_mask_labels(info, t)
        mask_prev = self._read_mask_labels(info, t - 1)

        if self.camera_crop and si in self._crop_cache:
            x0, y0, x1, y1 = self._crop_cache[si]
            img = img[y0:y1, x0:x1, :]
            mask_t = mask_t[y0:y1, x0:x1]
            mask_prev = mask_prev[y0:y1, x0:x1]

        img = resize_img(img, self.out_hw, self.resize_mode)
        mask_t = resize_mask(mask_t, self.out_hw, self.resize_mode)
        mask_prev = resize_mask(mask_prev, self.out_hw, self.resize_mode)

        if self.augment:
            img, mask_t, mask_prev = apply_augment(img, mask_t, mask_prev)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask_t.astype(np.int64))
        prev = torch.from_numpy(mask_prev.astype(np.int64))

        prev_oh = to_one_hot(prev, 3)
        v_oh = view_one_hot(view_idx, img_t.shape[1], img_t.shape[2])
        x = torch.cat([img_t, prev_oh, v_oh], dim=0)
        return x, y, view_idx


# ----------------------------
# Model: U-Net (GroupNorm)
# ----------------------------

def gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 3, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))

        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv1 = DoubleConv(base * 16, base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv2 = DoubleConv(base * 8, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv3 = DoubleConv(base * 4, base * 2)

        self.up4 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv4 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, num_classes, 1)

    @staticmethod
    def _pad_to(x, ref):
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)
        return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5)
        u1 = self._pad_to(u1, x4)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = self._pad_to(u2, x3)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = self._pad_to(u3, x2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = self._pad_to(u4, x1)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv4(u4)

        return self.outc(u4)


# ----------------------------
# Losses / metrics
# ----------------------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if self.ignore_bg:
            probs = probs[:, 1:, :, :]
            target_oh = target_oh[:, 1:, :, :]
        dims = (0, 2, 3)
        inter = torch.sum(probs * target_oh, dims)
        denom = torch.sum(probs + target_oh, dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


@torch.no_grad()
def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ----------------------------
# Training
# ----------------------------

@dataclass
class TrainConfig:
    manifest: Path
    out_dir: Path
    image_size: int = 512
    resize_mode: str = "stretch"
    batch_size: int = 6
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    seed: int = 123
    val_split: float = 0.2
    device: str = "cuda"
    base_channels: int = 32
    ce_bg_weight: float = 0.3
    amp: bool = True
    grad_accum: int = 1
    save_every: int = 1

    # GT colors (only used if masks are color-coded)
    mouse0_bgr: Tuple[int, int, int] = (25, 35, 40)
    mouse1_bgr: Tuple[int, int, int] = (40, 35, 25)
    mask_color_tol: int = 0

    # oversampling from frame-wise ethogram
    oversample_signals: Optional[List[str]] = None
    oversample_factor: float = 3.0

    # camera wiring
    camera_assert: bool = True
    camera_crop: bool = False
    camera_crop_margin: int = 10


def split_by_sample_id(items: List[SampleInfo], val_frac: float, seed: int) -> Tuple[List[SampleInfo], List[SampleInfo]]:
    rng = random.Random(seed)
    ids = sorted({x.sample_id for x in items})
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(ids[:n_val])
    train = [x for x in items if x.sample_id not in val_ids]
    val = [x for x in items if x.sample_id in val_ids]
    return train, val


def build_class_weights(bg_weight: float = 0.3) -> torch.Tensor:
    return torch.tensor([bg_weight, 1.0, 1.0], dtype=torch.float32)


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, scaler: GradScaler, epoch: int) -> None:
    ensure_dir(path.parent)
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch},
        path,
    )


def load_checkpoint(path: Path, model: nn.Module):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return int(ckpt.get("epoch", 0))


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    items = load_manifest(cfg.manifest)

    if cfg.camera_assert:
        for it in items:
            if it.camera_path is not None:
                assert_matches_camera(it.view, it.rgb_path, it.camera_path)

    train_items, val_items = split_by_sample_id(items, cfg.val_split, cfg.seed)

    gt_spec = MaskColorSpec(mouse0_bgr=cfg.mouse0_bgr, mouse1_bgr=cfg.mouse1_bgr, tol=int(cfg.mask_color_tol))
    out_hw = (cfg.image_size, cfg.image_size)

    train_ds = MicePairDataset(
        train_items,
        out_hw=out_hw,
        resize_mode=cfg.resize_mode,
        augment=True,
        gt_color_spec=gt_spec,
        oversample_signals=cfg.oversample_signals,
        oversample_factor=cfg.oversample_factor,
        min_t=1,
        camera_crop=cfg.camera_crop,
        camera_crop_margin=cfg.camera_crop_margin,
    )
    val_ds = MicePairDataset(
        val_items,
        out_hw=out_hw,
        resize_mode=cfg.resize_mode,
        augment=False,
        gt_color_spec=gt_spec,
        oversample_signals=None,          # do not oversample validation
        oversample_factor=1.0,
        min_t=1,
        camera_crop=cfg.camera_crop,
        camera_crop_margin=cfg.camera_crop_margin,
    )

    sampler = None
    shuffle = True
    if cfg.oversample_signals:
        sampler = WeightedRandomSampler(train_ds.weights, num_samples=len(train_ds.weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=9, num_classes=3, base=cfg.base_channels).to(device)

    ce_w = build_class_weights(cfg.ce_bg_weight).to(device)
    ce = nn.CrossEntropyLoss(weight=ce_w)
    dice = DiceLoss(num_classes=3, ignore_bg=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(cfg.amp, device)

    best_val = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, running_iou, n_batches = 0.0, 0.0, 0

        optim.zero_grad(set_to_none=True)

        for step, (x, y, view_idx) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast_ctx(cfg.amp, device):
                logits = model(x)
                loss = ce(logits, y) + dice(logits, y)

            scaler.scale(loss / cfg.grad_accum).backward()

            if step % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                running_iou += mean_iou(pred, y, num_classes=3)

            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_iou = running_iou / max(1, n_batches)

        model.eval()
        val_loss_sum, val_iou_sum, val_batches = 0.0, 0.0, 0
        per_view: Dict[int, List[float]] = {0: [], 1: [], 2: []}

        with torch.no_grad():
            for x, y, view_idx in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                view_idx_list = view_idx.cpu().numpy().tolist()

                with autocast_ctx(cfg.amp, device):
                    logits = model(x)
                    loss = ce(logits, y) + dice(logits, y)

                pred = torch.argmax(logits, dim=1)
                iou = mean_iou(pred, y, num_classes=3)

                val_loss_sum += float(loss.item())
                val_iou_sum += float(iou)
                val_batches += 1

                for bi, v in enumerate(view_idx_list):
                    per_view[int(v)].append(mean_iou(pred[bi:bi+1], y[bi:bi+1], num_classes=3))

        val_loss = val_loss_sum / max(1, val_batches)
        val_iou = val_iou_sum / max(1, val_batches)

        pv_str = " | ".join(
            f"{IDX_TO_VIEW[k]}:{(sum(vals)/len(vals)) if vals else 0.0:.3f}"
            for k, vals in per_view.items()
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_mIoU={train_iou:.3f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_iou:.3f} | {pv_str}"
        )

        if epoch % cfg.save_every == 0:
            save_checkpoint(cfg.out_dir / f"ckpt_epoch_{epoch:03d}.pt", model, optim, scaler, epoch)
        if val_iou > best_val:
            best_val = val_iou
            save_checkpoint(cfg.out_dir / "ckpt_best.pt", model, optim, scaler, epoch)


# ----------------------------
# Prediction
# ----------------------------

def colorize_mask_rgb(mask_hw: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    out[mask_hw == 1] = (255, 0, 0)   # RGB
    out[mask_hw == 2] = (0, 255, 0)
    return out


@torch.no_grad()
def predict_sequence(
    model: nn.Module,
    rgb_source: Path,
    view: str,
    out_dir: Path,
    image_size: int,
    resize_mode: str,
    device: str,
    save_overlay: bool,
    alpha: float,
    out_mask_video: Optional[Path],
    save_fullres_masks: bool,
    mouse0_bgr: Tuple[int, int, int],
    mouse1_bgr: Tuple[int, int, int],
    fps_override: Optional[float],
    cameras_json: Optional[Path],
    camera_crop: bool,
    camera_crop_margin: int,
):
    ensure_dir(out_dir)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(device_t)

    view = view.lower().strip()
    view_idx = VIEWS[view]

    crop_box = None
    if cameras_json is not None:
        assert_matches_camera(view, rgb_source, cameras_json)
        if camera_crop:
            crop_box = cage_crop_box(cameras_json, view, margin=camera_crop_margin)

    src_is_dir = rgb_source.is_dir()
    if src_is_dir:
        frames = list_frames(rgb_source)
        n = len(frames)

        def get_rgb(i: int) -> np.ndarray:
            return read_image_rgb(frames[i])

        src_fps = float(fps_override) if fps_override is not None else 30.0
    else:
        vr = VideoReader(rgb_source)
        n = len(vr)

        def get_rgb(i: int) -> np.ndarray:
            return vr.read_rgb(i)

        src_fps = float(fps_override) if fps_override is not None else vr.fps()

    prev_probs = torch.zeros(1, 3, image_size, image_size, device=device_t, dtype=torch.float32)
    prev_probs[:, 0, :, :] = 1.0

    vw = None

    for t in range(n):
        img_full = get_rgb(t)
        H0, W0 = img_full.shape[:2]

        if out_mask_video is not None and vw is None:
            ensure_dir(out_mask_video.parent)
            vw = cv2.VideoWriter(str(out_mask_video), fourcc_for_path(out_mask_video), src_fps, (W0, H0), True)
            if not vw.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {out_mask_video}")

        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            img_in = img_full[y0:y1, x0:x1, :]
            Hc, Wc = img_in.shape[:2]
        else:
            img_in = img_full
            x0 = y0 = 0
            x1, y1 = W0, H0
            Hc, Wc = H0, W0

        img_model = resize_img(img_in, (image_size, image_size), resize_mode)

        img_t = torch.from_numpy(img_model).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        v_oh = view_one_hot(view_idx, image_size, image_size).unsqueeze(0)
        x = torch.cat([img_t.to(device_t), prev_probs, v_oh.to(device_t)], dim=1)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_model = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        prev_probs = probs.detach()

        # Optionally map prediction back to full-resolution frame coordinates (no distortion).
        # Needed for: (a) full-res mask export, (b) composited output video.
        full_labels = None
        if save_fullres_masks or (out_mask_video is not None):
            if resize_mode == "letterbox":
                nw, nh, pad_x, pad_y = letterbox_params(Wc, Hc, image_size, image_size)
                pred_core = pred_model[pad_y:pad_y + nh, pad_x:pad_x + nw]
                pred_crop = cv2.resize(pred_core, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            else:
                pred_crop = cv2.resize(pred_model, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            full_labels = np.zeros((H0, W0), dtype=np.uint8)
            full_labels[y0:y1, x0:x1] = pred_crop

        # Always save the model-resolution mask (image_size x image_size)
        cv2.imwrite(str(out_dir / f"mask_{t:06d}.png"), pred_model)
        # Optionally save full-resolution label mask aligned with the input frame
        if save_fullres_masks:
            assert full_labels is not None
            cv2.imwrite(str(out_dir / f"mask_full_{t:06d}.png"), full_labels)

        if save_overlay:
            col = colorize_mask_rgb(pred_model)
            overlay = (1 - alpha) * img_model.astype(np.float32) + alpha * col.astype(np.float32)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"overlay_{t:06d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if vw is not None:
            assert full_labels is not None
            out_bgr = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            out_bgr[full_labels == 1] = mouse0_bgr
            out_bgr[full_labels == 2] = mouse1_bgr
            vw.write(out_bgr)

    if not src_is_dir:
        vr.close()
    if vw is not None:
        vw.release()


# ----------------------------
# Extraction utility
# ----------------------------

def extract_frames(
    rgb_video: Path,
    mask_video: Optional[Path],
    out_rgb_dir: Path,
    out_mask_dir: Optional[Path],
    every: int,
    color_spec: MaskColorSpec,
):
    ensure_dir(out_rgb_dir)
    if out_mask_dir is not None:
        ensure_dir(out_mask_dir)

    vr = VideoReader(rgb_video)
    n = len(vr)

    mr = None
    if mask_video is not None:
        mr = VideoReader(mask_video)
        n = min(n, len(mr))

    for i in range(0, n, every):
        img_rgb = vr.read_rgb(i)
        cv2.imwrite(str(out_rgb_dir / f"{i:06d}.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        if mr is not None and out_mask_dir is not None:
            mask_bgr = mr.read_bgr(i)
            labels = decode_mask_to_labels(mask_bgr, color_spec)
            cv2.imwrite(str(out_mask_dir / f"{i:06d}.png"), labels)

    vr.close()
    if mr is not None:
        mr.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract video(s) to frame folders.")
    pe.add_argument("--rgb_video", type=Path, required=True)
    pe.add_argument("--mask_video", type=Path, default=None)
    pe.add_argument("--out_rgb_dir", type=Path, required=True)
    pe.add_argument("--out_mask_dir", type=Path, default=None)
    pe.add_argument("--every", type=int, default=1)
    pe.add_argument("--mouse0-bgr", type=parse_bgr, default=parse_bgr("25,35,40"))
    pe.add_argument("--mouse1-bgr", type=parse_bgr, default=parse_bgr("40,35,25"))
    pe.add_argument("--mask-color-tol", type=int, default=0)

    pt = sub.add_parser("train", help="Train the segmentation model.")
    pt.add_argument("--manifest", type=Path, required=True)
    pt.add_argument("--out_dir", type=Path, required=True)
    pt.add_argument("--image_size", type=int, default=512)
    pt.add_argument("--resize_mode", type=str, default="stretch", choices=["stretch", "letterbox"])
    pt.add_argument("--batch_size", type=int, default=6)
    pt.add_argument("--num_workers", type=int, default=4)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--seed", type=int, default=123)
    pt.add_argument("--val_split", type=float, default=0.2)
    pt.add_argument("--device", type=str, default="cuda")
    pt.add_argument("--base_channels", type=int, default=32)
    pt.add_argument("--ce_bg_weight", type=float, default=0.3)
    pt.add_argument("--no_amp", action="store_true")
    pt.add_argument("--grad_accum", type=int, default=1)
    pt.add_argument("--save_every", type=int, default=1)

    pt.add_argument("--mouse0-bgr", type=parse_bgr, default=parse_bgr("25,35,40"))
    pt.add_argument("--mouse1-bgr", type=parse_bgr, default=parse_bgr("40,35,25"))
    pt.add_argument("--mask-color-tol", type=int, default=0)

    pt.add_argument("--oversample_signals", type=str, default="",
                    help="Comma-separated ethogram columns to oversample when ==1 (e.g. interaction,mouse0_rear,mouse1_rear).")
    pt.add_argument("--oversample_factor", type=float, default=3.0)

    pt.add_argument("--no_camera_assert", action="store_true")
    pt.add_argument("--camera_crop", action="store_true")
    pt.add_argument("--camera_crop_margin", type=int, default=10)

    pp = sub.add_parser("predict", help="Inference with autoregressive prev-mask feedback.")
    pp.add_argument("--ckpt", type=Path, required=True)
    pp.add_argument("--rgb_source", type=Path, required=True)
    pp.add_argument("--view", type=str, required=True, choices=list(VIEWS.keys()))
    pp.add_argument("--out_dir", type=Path, required=True)
    pp.add_argument("--image_size", type=int, default=512)
    pp.add_argument("--resize_mode", type=str, default="stretch", choices=["stretch", "letterbox"])
    pp.add_argument("--device", type=str, default="cuda")
    pp.add_argument("--no_overlay", action="store_true")
    pp.add_argument("--alpha", type=float, default=0.4)

    pp.add_argument("--out_mask_video", type=Path, default=None)
    pp.add_argument("--save_fullres_masks", action="store_true",
                    help="Also write full-resolution label masks aligned to the input frames as mask_full_XXXXXX.png.")
    pp.add_argument("--mouse0-bgr", type=parse_bgr, default=parse_bgr("40,35,255"))
    pp.add_argument("--mouse1-bgr", type=parse_bgr, default=parse_bgr("255,255,40"))
    pp.add_argument("--fps", type=float, default=None)

    pp.add_argument("--cameras_json", type=Path, default=None)
    pp.add_argument("--camera_crop", action="store_true")
    pp.add_argument("--camera_crop_margin", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "extract":
        spec = MaskColorSpec(mouse0_bgr=args.mouse0_bgr, mouse1_bgr=args.mouse1_bgr, tol=int(args.mask_color_tol))
        extract_frames(
            rgb_video=args.rgb_video,
            mask_video=args.mask_video,
            out_rgb_dir=args.out_rgb_dir,
            out_mask_dir=args.out_mask_dir,
            every=int(args.every),
            color_spec=spec,
        )
        print("Done extracting.")

    elif args.cmd == "train":
        signals = [s.strip() for s in args.oversample_signals.split(",") if s.strip()]
        cfg = TrainConfig(
            manifest=args.manifest,
            out_dir=args.out_dir,
            image_size=args.image_size,
            resize_mode=args.resize_mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=args.seed,
            val_split=args.val_split,
            device=args.device,
            base_channels=args.base_channels,
            ce_bg_weight=args.ce_bg_weight,
            amp=(not args.no_amp),
            grad_accum=max(1, args.grad_accum),
            save_every=max(1, args.save_every),
            mouse0_bgr=args.mouse0_bgr,
            mouse1_bgr=args.mouse1_bgr,
            mask_color_tol=int(args.mask_color_tol),
            oversample_signals=signals if signals else None,
            oversample_factor=float(args.oversample_factor),
            camera_assert=(not args.no_camera_assert),
            camera_crop=bool(args.camera_crop),
            camera_crop_margin=int(args.camera_crop_margin),
        )
        train(cfg)

    elif args.cmd == "predict":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=9, num_classes=3, base=32)
        load_checkpoint(args.ckpt, model)

        predict_sequence(
            model=model,
            rgb_source=args.rgb_source,
            view=args.view,
            out_dir=args.out_dir,
            image_size=int(args.image_size),
            resize_mode=args.resize_mode,
            device=str(device),
            save_overlay=(not args.no_overlay),
            alpha=float(args.alpha),
            out_mask_video=args.out_mask_video,
            save_fullres_masks=bool(args.save_fullres_masks),
            mouse0_bgr=args.mouse0_bgr,
            mouse1_bgr=args.mouse1_bgr,
            fps_override=args.fps,
            cameras_json=args.cameras_json,
            camera_crop=bool(args.camera_crop),
            camera_crop_margin=int(args.camera_crop_margin),
        )
        print("Done predicting.")


if __name__ == "__main__":
    main()
