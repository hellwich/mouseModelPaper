#!/usr/bin/env python3
"""
extract_only.py

Standalone extractor that DOES NOT import torch.
- Extracts RGB video to frames
- Extracts mask video (either indexed 0/1/2 or BGR color-coded) to single-channel label PNGs (0/1/2)

Supports .mp4/.mkv etc via OpenCV.

Usage example:
  python extract_only.py \
    --rgb_video cam1_top.mp4 \
    --mask_video cam1_top_seg_id.mkv \
    --out_rgb_dir frames/vid001_top_rgb \
    --out_mask_dir frames/vid001_top_mask \
    --every 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    c0 = mask_bgr[:, :, 0]
    if not (np.array_equal(c0, mask_bgr[:, :, 1]) and np.array_equal(c0, mask_bgr[:, :, 2])):
        return None
    u = np.unique(c0)
    if u.size and (u.min() < 0 or u.max() > 2):
        return None
    return c0.astype(np.uint8)


def decode_mask_to_labels(mask_arr: np.ndarray, mouse0_bgr: Tuple[int,int,int], mouse1_bgr: Tuple[int,int,int], tol: int) -> np.ndarray:
    """Return HxW uint8 labels in {0:bg,1:mouse0,2:mouse1}."""
    if mask_arr.ndim == 2:
        raw = mask_arr.astype(np.int64)
        u = np.unique(raw)
        if u.size and (u.min() < 0 or u.max() > 2):
            raise ValueError(f"Indexed mask has values outside 0..2: {u[:10].tolist()}")
        return raw.astype(np.uint8)

    if mask_arr.ndim == 3 and mask_arr.shape[2] in (3, 4):
        if mask_arr.shape[2] == 4:
            mask_arr = mask_arr[:, :, :3]
        mask_bgr = mask_arr

        maybe = _try_decode_indexed_from_3ch(mask_bgr)
        if maybe is not None:
            return maybe

        bg = _match_color(mask_bgr, (0,0,0), tol)
        m0 = _match_color(mask_bgr, mouse0_bgr, tol)
        m1 = _match_color(mask_bgr, mouse1_bgr, tol)

        out = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
        out[m0] = 1
        out[m1] = 2

        unknown = ~(bg | m0 | m1)
        if np.any(unknown):
            bad_pixels = mask_bgr[unknown]
            uniq = np.unique(bad_pixels.reshape(-1, 3), axis=0)[:10]
            raise ValueError(
                "Mask contains colors not matching indexed 0/1/2 OR "
                f"bg=(0,0,0), mouse0={mouse0_bgr}, mouse1={mouse1_bgr}. "
                f"First unknown BGR colors: {uniq.tolist()} (tol={tol})"
            )
        return out

    raise ValueError(f"Unsupported mask shape: {mask_arr.shape}")


def extract(rgb_video: Path, mask_video: Optional[Path], out_rgb_dir: Path, out_mask_dir: Optional[Path],
            every: int, mouse0_bgr: Tuple[int,int,int], mouse1_bgr: Tuple[int,int,int], tol: int):
    ensure_dir(out_rgb_dir)
    if out_mask_dir is not None:
        ensure_dir(out_mask_dir)

    cap_rgb = cv2.VideoCapture(str(rgb_video))
    if not cap_rgb.isOpened():
        raise FileNotFoundError(f"Cannot open rgb_video: {rgb_video}")

    cap_mask = None
    if mask_video is not None:
        cap_mask = cv2.VideoCapture(str(mask_video))
        if not cap_mask.isOpened():
            raise FileNotFoundError(f"Cannot open mask_video: {mask_video}")

    i = 0
    saved = 0
    while True:
        ok, frame_bgr = cap_rgb.read()
        if not ok or frame_bgr is None:
            break

        mask_frame = None
        if cap_mask is not None:
            okm, mask_bgr = cap_mask.read()
            if not okm or mask_bgr is None:
                break
            mask_frame = mask_bgr

        if i % every == 0:
            cv2.imwrite(str(out_rgb_dir / f"{i:06d}.png"), frame_bgr)
            if out_mask_dir is not None and mask_frame is not None:
                labels = decode_mask_to_labels(mask_frame, mouse0_bgr, mouse1_bgr, tol)
                cv2.imwrite(str(out_mask_dir / f"{i:06d}.png"), labels)
            saved += 1
        i += 1

    cap_rgb.release()
    if cap_mask is not None:
        cap_mask.release()

    print(f"Done. Read frames={i}, saved={saved} (every={every}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_video", type=Path, required=True)
    ap.add_argument("--mask_video", type=Path, default=None)
    ap.add_argument("--out_rgb_dir", type=Path, required=True)
    ap.add_argument("--out_mask_dir", type=Path, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--mouse0-bgr", type=parse_bgr, default=parse_bgr("25,35,40"))
    ap.add_argument("--mouse1-bgr", type=parse_bgr, default=parse_bgr("40,35,25"))
    ap.add_argument("--mask-color-tol", type=int, default=0)
    args = ap.parse_args()

    extract(
        rgb_video=args.rgb_video,
        mask_video=args.mask_video,
        out_rgb_dir=args.out_rgb_dir,
        out_mask_dir=args.out_mask_dir,
        every=int(args.every),
        mouse0_bgr=args.mouse0_bgr,
        mouse1_bgr=args.mouse1_bgr,
        tol=int(args.mask_color_tol),
    )


if __name__ == "__main__":
    main()
