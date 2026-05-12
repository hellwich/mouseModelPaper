#!/usr/bin/env python3
"""Batch inference for a trained DeepLabCut multi-animal project (two mice).

What it does:
- Finds camera videos (default: cam*.mp4) inside a directory.
- Runs deeplabcut.analyze_videos with auto_track=True (pose + tracking) to produce per-frame 2D keypoints.
- Optionally renders labeled videos for quick visual QC.

This script is intentionally "version-tolerant": it only passes kwargs that exist in
your installed deeplabcut version.

Typical usage:
  conda activate dlc_mouse
  python predict_dlc_multi_mouse_batch.py \
    --config /path/to/dlc_project/config.yaml \
    --videos-dir /path/to/render_out \
    --destfolder /path/to/render_out/dlc_predictions \
    --n-tracks 2 \
    --create-labeled-video
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_call(func, *args, **kwargs):
    """Call DLC API with only supported kwargs for the installed version."""
    sig = inspect.signature(func)
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported and v is not None}
    dropped = sorted([k for k in kwargs.keys() if k not in supported and kwargs[k] is not None])
    if dropped:
        print(f"[compat] {func.__name__}: dropped unsupported kwargs: {dropped}")
    return func(*args, **filtered)


def _find_videos(videos_dir: Path, pattern: str) -> List[Path]:
    vids = sorted(videos_dir.glob(pattern))
    # also accept nested common layout
    if not vids:
        vids = sorted((videos_dir / "videos").glob(pattern))
    return [p.resolve() for p in vids if p.is_file()]


def _ensure_config_has_batch_size(config_path: Path, batchsize: int) -> bool:
    """DLC TF analyze_videos (2.3.x) may expect cfg['batch_size'] in project config.

    If missing, we add it. This is a tiny, safe augmentation.

    Returns True if file was modified.
    """
    try:
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with config_path.open("r") as f:
            cfg = yaml.load(f)
        if cfg is None:
            return False
        if "batch_size" not in cfg:
            cfg["batch_size"] = int(batchsize)
            with config_path.open("w") as f:
                yaml.dump(cfg, f)
            return True
        return False
    except Exception:
        # If ruamel isn't available or parsing fails, don't hard-fail; rely on passing batchsize kw.
        return False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run trained DLC (multi-animal) on a folder of videos.")
    p.add_argument("--config", type=str, required=True, help="Path to DLC project config.yaml")
    p.add_argument("--videos-dir", type=str, required=True, help="Directory containing rendered videos")
    p.add_argument("--destfolder", type=str, default=None, help="Where to write DLC outputs (default: <videos-dir>/dlc_predictions)")

    p.add_argument("--pattern", type=str, default="cam*.mp4", help="Glob pattern for videos inside videos-dir")
    p.add_argument("--videotype", type=str, default=".mp4", help="Video extension filter passed to DLC")

    p.add_argument("--shuffle", type=int, default=1, help="Shuffle index to use (usually 1)")
    p.add_argument("--snapshot", type=str, default=None, help="Optional snapshot to use, e.g. 'snapshot-200000'")

    # Key fix for your crash:
    p.add_argument("--batchsize", type=int, default=8, help="Inference batch size passed to DLC (and optionally written into config if missing)")
    p.add_argument("--patch-config-batchsize", action="store_true", help="If config.yaml lacks 'batch_size', add it before inference")

    p.add_argument("--gputouse", type=int, default=0, help="GPU index for TensorFlow inference")

    # Multi-animal tracking options
    p.add_argument("--auto-track", action="store_true", default=True, help="Use DLC auto_track pipeline (pose + tracking)")
    p.add_argument("--no-auto-track", dest="auto_track", action="store_false", help="Disable auto_track")
    p.add_argument("--n-tracks", type=int, default=2, help="Number of animals to track (2 mice)")

    p.add_argument("--save-as-csv", action="store_true", default=True, help="Also save CSV outputs (in addition to H5)")
    p.add_argument("--no-csv", dest="save_as_csv", action="store_false", help="Do not save CSV")

    p.add_argument("--create-labeled-video", action="store_true", help="Create labeled videos for QC")
    p.add_argument("--plot-trajectories", action="store_true", help="Plot trajectories")

    # Not all DLC versions accept these; safe_call will drop if unsupported.
    p.add_argument("--animal-names", type=str, default=None, help="Comma-separated individual names (optional); e.g. mouse0,mouse1")
    p.add_argument("--engine", type=str, default="pytorch", choices=["pytorch", "tensorflow", "auto"], help="Requested engine (compat; may be ignored by DLC)")
    p.add_argument("--reid", action="store_true", help="If supported, enable reID/tracking transformer")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    config = Path(args.config).expanduser().resolve()
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    if not videos_dir.exists():
        raise SystemExit(f"videos-dir not found: {videos_dir}")

    destfolder = Path(args.destfolder).expanduser().resolve() if args.destfolder else (videos_dir / "dlc_predictions")
    destfolder.mkdir(parents=True, exist_ok=True)

    video_paths = _find_videos(videos_dir, args.pattern)
    if not video_paths:
        raise SystemExit(f"No videos found in {videos_dir} matching pattern '{args.pattern}'")

    print(f"Using DLC config: {config}")
    print(f"Requested engine: {args.engine}")
    print(f"Videos ({len(video_paths)}):")
    for p in video_paths:
        print(f"  - {p}")
    print(f"DLC outputs will be written to: {destfolder}")

    # Optional config patch (fixes KeyError: 'batch_size' for some DLC builds)
    if args.patch_config_batchsize:
        changed = _ensure_config_has_batch_size(config, args.batchsize)
        if changed:
            print(f"[patch] Added batch_size: {args.batchsize} to {config}")

    # Import DLC late so env vars/hooks are applied
    try:
        import deeplabcut as dlc
    except Exception as e:
        raise SystemExit(f"Failed to import deeplabcut in this environment: {e}")

    # Analyze
    analyze_kwargs: Dict[str, Any] = dict(
        videotype=args.videotype,
        shuffle=args.shuffle,
        save_as_csv=bool(args.save_as_csv),
        destfolder=str(destfolder),
        gputouse=int(args.gputouse),
        TFGPUinference=True,
        auto_track=bool(args.auto_track),
        n_tracks=int(args.n_tracks),
        # Important: some DLC versions use 'batchsize' (not 'batch_size')
        batchsize=int(args.batchsize),
        # For newer versions, but safe_call will drop if unsupported:
        batch_size=int(args.batchsize),
        engine=None if args.engine == "auto" else args.engine,
        animal_names=None if not args.animal_names else [s.strip() for s in args.animal_names.split(",") if s.strip()],
        snapshot=args.snapshot,
        reid=bool(args.reid),
    )

    print("[1/3] Analyzing videos (pose + tracking)...")
    try:
        _safe_call(dlc.analyze_videos, str(config), [str(p) for p in video_paths], **analyze_kwargs)
    except KeyError as ke:
        # Common DLC 2.3.x gotcha: cfg['batch_size'] missing
        if str(ke).strip("'") == "batch_size":
            print("\n[error] DeepLabCut raised KeyError('batch_size') during analyze_videos.")
            print("This typically means your project config.yaml lacks a 'batch_size' entry and your DLC build expects it.")
            print("Fix options:")
            print(f"  1) Re-run with --patch-config-batchsize (already default in this script)\n")
            print(f"  2) Manually add to config.yaml: batch_size: {args.batchsize}\n")
            print(f"  3) Ensure analyze_videos receives a batchsize kwarg (this script does).\n")
            raise
        raise

    # Optional labeled videos
    if args.create_labeled_video:
        print("[2/3] Creating labeled videos...")
        _safe_call(
            dlc.create_labeled_video,
            str(config),
            [str(p) for p in video_paths],
            videotype=args.videotype,
            shuffle=args.shuffle,
            destfolder=str(destfolder),
        )

    if args.plot_trajectories:
        print("[3/3] Plotting trajectories...")
        _safe_call(
            dlc.plot_trajectories,
            str(config),
            [str(p) for p in video_paths],
            videotype=args.videotype,
            shuffle=args.shuffle,
            destfolder=str(destfolder),
        )

    print("Done.")


if __name__ == "__main__":
    main()
