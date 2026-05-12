#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

# Make both the python/ folder (mice3d package) and repo root (ceres_point_ba .so) importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../python
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # repo root
for p in (THIS_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mice3d.cameras import load_cameras_json
from mice3d.dlc import load_dlc_long_csv
from mice3d.pipeline import Mouse3DReconstructor, ReconstructionParams
from mice3d.segmentation import SegmentationDirectorySource, SegmentationVideoSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-view mouse triangulation + identity cleaning")
    p.add_argument("--cameras", required=True, help="Path to cameras.json")
    p.add_argument("--dlc", required=True, help="Path to dlc_long.csv")
    p.add_argument(
        "--out",
        required=True,
        help="Output prefix (writes *_triangulated3d.csv and *_dlc_assignments.csv)",
    )

    seg = p.add_mutually_exclusive_group(required=False)
    seg.add_argument(
        "--seg-dir",
        help=(
            "Root directory containing masks. Auto-detects common layouts like "
            "<root>/<cam_name>/frame_000000.png or <root>/top/mask_000001.png"
        ),
    )
    seg.add_argument(
        "--seg-videos",
        nargs=3,
        metavar=("TOP", "FRONT", "SIDE"),
        help="Three segmentation videos in order: cam1_top cam2_front cam3_side",
    )

    # Common params
    p.add_argument("--likelihood-min", type=float, default=0.0)

    # Stage A thresholds
    p.add_argument("--accept-rms", type=float, default=6.0, help="Accept BA rms_px for triplets")
    p.add_argument("--accept-rms-pair", type=float, default=6.0, help="Accept BA rms_px for pairs")
    p.add_argument("--e-init", type=float, default=12.0, help="Fast reprojection gate (px)")
    p.add_argument("--d-max", type=float, default=20.0, help="Ray closest-approach gate (mm)")
    p.add_argument("--theta-trip", type=float, default=1.5, help="Min ray angle triplet (deg)")
    p.add_argument("--theta-pair", type=float, default=1.0, help="Min ray angle pair (deg)")

    # Stage B segmentation window
    p.add_argument("--seg-radius", type=int, default=9)
    p.add_argument("--seg-sigma", type=float, default=4.0)
    p.add_argument("--vote-thr", type=float, default=0.01)

    # Optional Stage B debug
    p.add_argument(
        "--debug_stageB",
        "--debug-stageB",
        action="store_true",
        dest="debug_stageB",
        help="Verbose Stage B debug prints (seg votes + homogenization decisions)",
    )

    # Optional Stage B graph plausibility model (B7)
    p.add_argument(
        "--graph_model",
        "--graph-model",
        default=None,
        help="Path to skeleton graph model (enables Stage B7 leaf plausibility check)",
    )

    p.add_argument(
        "--graph-rigid-tol-mm-max",
        type=float,
        default=None,
        help=(
            "(B7) Max allowed absolute deviation (mm) between an estimated leaf 3D point "
            "and the rigidly-transformed template leaf position. If omitted, use the code default."
        ),
    )
    p.add_argument(
        "--graph-rigid-tol-ratio",
        type=float,
        default=None,
        help=(
            "(B7) Optional additional tolerance scaling by template edge length: tol=min(mm_max, ratio*L). "
            "If omitted, use the code default."
        ),
    )

    p.add_argument(
        "--frames",
        "--frame",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Process frame range [START, END) (default all)",
    )

    # Debugging / diagnostics
    p.add_argument("--debug", action="store_true", help="Print per-frame summaries")
    p.add_argument("--debug-save", action="store_true", help="Save per-frame debug JSONL next to output prefix")
    p.add_argument(
        "--debug_stageA",
        "--debug-stageA",
        action="store_true",
        dest="debug_stageA",
        help=(
            "Print every Stage A hypothesis tested, including per-gate pass/fail diagnostics and BA outcome. "
            "(Very verbose)"
        ),
    )
    p.add_argument("--faulthandler", action="store_true", help="Enable Python faulthandler (useful for segfaults)")
    p.add_argument("--verbose-ceres", action="store_true", help="Let Ceres print iteration progress")
    p.add_argument("--trace-ceres", action="store_true", help="Print inputs to each Ceres solve (very verbose)")

    # Segmentation backend (opencv can segfault on some setups)
    p.add_argument(
        "--seg-backend",
        choices=["imageio", "opencv"],
        default="imageio",
        help="How to read segmentation masks/videos",
    )

    # Optional overrides for segmentation directory layouts
    p.add_argument(
        "--seg-ext",
        default=None,
        help="Override mask file extension (e.g. png). If omitted, infer from directory.",
    )
    p.add_argument(
        "--seg-prefix",
        default=None,
        help="Override filename prefix (e.g. mask or frame). If omitted, infer from directory.",
    )
    p.add_argument(
        "--seg-pad",
        type=int,
        default=None,
        help="Override zero-padding width (e.g. 6 for 000001). If omitted, infer.",
    )
    p.add_argument(
        "--seg-start-index",
        type=int,
        default=None,
        help=(
            "Override start index used to map frame->file index. "
            "Example: if files start at mask_000001.png for frame 0, set to 1. "
            "If omitted, infer from min index in directory."
        ),
    )
    p.add_argument(
        "--seg-cam-map",
        action="append",
        default=[],
        metavar="CAM=DIR",
        help=(
            "Override camera subdir mapping (repeatable). Example: "
            "--seg-cam-map cam1_top=top --seg-cam-map cam2_front=front"
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.faulthandler:
        import faulthandler

        faulthandler.enable()

    rig = load_cameras_json(args.cameras)
    dlc = load_dlc_long_csv(args.dlc, camera_order=[c.name for c in rig.cameras])

    seg_source = None
    if args.seg_dir:
        cam_dir_map = {}
        for s in args.seg_cam_map:
            if "=" not in s:
                raise SystemExit(f"Bad --seg-cam-map value '{s}'. Use CAM=DIR.")
            k, v = s.split("=", 1)
            cam_dir_map[k.strip()] = v.strip()

        seg_source = SegmentationDirectorySource(
            args.seg_dir,
            ext=args.seg_ext,
            backend=args.seg_backend,
            cam_names=[c.name for c in rig.cameras],
            cam_dir_map=cam_dir_map if cam_dir_map else None,
            filename_prefix=args.seg_prefix,
            pad=args.seg_pad,
            start_index=args.seg_start_index,
        )
    elif args.seg_videos:
        video_paths: Dict[str, str] = {
            "cam1_top": args.seg_videos[0],
            "cam2_front": args.seg_videos[1],
            "cam3_side": args.seg_videos[2],
        }
        seg_source = SegmentationVideoSource(video_paths, backend=args.seg_backend)

    params = ReconstructionParams()
    params.likelihood_min = args.likelihood_min

    params.stage_a.accept_rms_px_triplet = args.accept_rms
    params.stage_a.accept_rms_px_pair = args.accept_rms_pair
    params.stage_a.e_init_px = args.e_init
    params.stage_a.d_max_mm = args.d_max
    params.stage_a.theta_min_triplet_deg = args.theta_trip
    params.stage_a.theta_min_pair_deg = args.theta_pair

    params.stage_a.bbox_margin_mm = 1000.0  # diagnostic: effectively disables bbox gate

    params.stage_a.verbose_ceres = bool(args.verbose_ceres)
    params.stage_a.trace_ceres_inputs = bool(args.trace_ceres)
    params.stage_a.debug_stageA = bool(getattr(args, "debug_stageA", False))

    params.stage_b.seg_radius = args.seg_radius
    params.stage_b.seg_sigma = args.seg_sigma
    params.stage_b.vote_thr = args.vote_thr

    # Stage B extras
    params.stage_b.debug_stageB = bool(getattr(args, "debug_stageB", False))
    if getattr(args, "graph_model", None):
        params.stage_b.graph_model = str(args.graph_model)

    if getattr(args, "graph_rigid_tol_mm_max", None) is not None:
        params.stage_b.graph_rigid_tol_mm_max = float(args.graph_rigid_tol_mm_max)
    if getattr(args, "graph_rigid_tol_ratio", None) is not None:
        params.stage_b.graph_rigid_tol_ratio = float(args.graph_rigid_tol_ratio)

    debug_path = (args.out + "_debug.jsonl") if args.debug_save else None
    recon = Mouse3DReconstructor(
        rig=rig,
        dlc=dlc,
        seg_source=seg_source,
        params=params,
        debug=bool(args.debug),
        debug_save_path=debug_path,
    )

    if args.frames:
        start, end = args.frames
        recon.run_all(frames=list(range(start, end)))
    else:
        recon.run_all()

    tri_path, ass_path = recon.save_outputs(args.out)
    print("Wrote:")
    print(" ", tri_path)
    print(" ", ass_path)
    if debug_path:
        print(" ", debug_path)


if __name__ == "__main__":
    main()
