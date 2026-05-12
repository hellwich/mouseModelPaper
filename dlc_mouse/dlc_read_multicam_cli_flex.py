#!/usr/bin/env python3
"""
dlc_read_multicam_cli.py

Standalone CLI utility to load multi-camera multi-animal DeepLabCut outputs
(e.g., *_el.h5) and export:

1) A long-format table (CSV/Parquet) with one row per:
   (frame, camera, individual, bodypart) -> x, y, likelihood

2) Optionally a dense NumPy array saved as NPZ:
   arr shape = (T, C, I, B, 3) with (x, y, likelihood)

Assumes DLC multi-animal H5 columns are a 4-level MultiIndex:
  (scorer, individual, bodypart, coord) where coord in {'x','y','likelihood'}.

Examples:
  python dlc_read_multicam_cli.py \
    --input cam1_top:/path/cam1_top*_el.h5 \
    --input cam2_front:/path/cam2_front*_el.h5 \
    --input cam3_side:/path/cam3_side*_el.h5 \
    --out-long /tmp/dlc_long.csv \
    --out-npz /tmp/dlc_array.npz

If you provide globs, the script will expand them; each camera must resolve to exactly
one file.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _read_first_dataframe_from_h5(h5_path: str | Path) -> pd.DataFrame:
    """Read the first DataFrame from an HDF5 file (handles varying keys)."""
    h5_path = str(h5_path)
    try:
        df = pd.read_hdf(h5_path)
        if isinstance(df, pd.DataFrame):
            return df
    except (ValueError, KeyError):
        pass

    with pd.HDFStore(h5_path, mode="r") as store:
        for key in store.keys():
            obj = store.get(key)
            if isinstance(obj, pd.DataFrame):
                return obj

    raise RuntimeError(f"No DataFrame found in H5 file: {h5_path}")


def _ensure_expected_multindex(
    df: pd.DataFrame,
    *,
    default_individual: str = "mouse0",
) -> pd.DataFrame:
    """
    Standardize DeepLabCut outputs to a 4-level MultiIndex with levels:

        (scorer, individual, bodypart, coord)

    DLC commonly writes:

    - single-animal projects: 3 levels (scorer, bodypart, coord)
    - multi-animal projects: 4 levels (scorer, individual, bodypart, coord)

    This function accepts both (and minor permutations) and returns a copy with
    standardized 4-level columns. For 3-level inputs, `default_individual` is
    inserted as the missing individual level.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            f"Expected MultiIndex columns from DLC. Got: {type(df.columns)}"
        )

    mi = df.columns
    want_coords = {"x", "y", "likelihood"}

    # Identify which level looks like the coordinate level.
    coord_levels = []
    for i in range(mi.nlevels):
        vals = {str(v) for v in mi.get_level_values(i).unique()}
        if want_coords.issubset(vals):
            coord_levels.append(i)
    if not coord_levels:
        raise ValueError(
            "Could not identify coord level. Expected one MultiIndex level to contain "
            f"{sorted(want_coords)}. nlevels={mi.nlevels}, level uniques="
            f"{[len(mi.get_level_values(i).unique()) for i in range(mi.nlevels)]}"
        )
    coord_level = coord_levels[-1]  # if multiple candidates, prefer the last

    remaining = [i for i in range(mi.nlevels) if i != coord_level]
    if mi.nlevels not in (3, 4):
        raise ValueError(
            f"Expected 3- or 4-level MultiIndex columns. Got nlevels={mi.nlevels}."
        )

    # Heuristic: scorer level typically has the smallest cardinality.
    uniq_counts = {i: len(mi.get_level_values(i).unique()) for i in remaining}
    scorer_level = min(remaining, key=lambda i: uniq_counts[i])

    if mi.nlevels == 3:
        bodypart_level = [i for i in remaining if i != scorer_level][0]
        scorer = mi.get_level_values(scorer_level).astype(str)
        bodypart = mi.get_level_values(bodypart_level).astype(str)
        coord = mi.get_level_values(coord_level).astype(str)
        individual = pd.Index([default_individual] * len(mi), dtype="object")

        out = df.copy()
        out.columns = pd.MultiIndex.from_arrays(
            [scorer, individual, bodypart, coord],
            names=["scorer", "individual", "bodypart", "coord"],
        )
        return out

    # mi.nlevels == 4: determine which remaining level is individual vs bodypart.
    # Heuristic: individual level usually has fewer uniques than bodyparts.
    rem2 = [i for i in remaining if i != scorer_level]
    if len(rem2) != 2:
        raise ValueError("Unexpected MultiIndex structure while standardizing columns.")
    individual_level = min(rem2, key=lambda i: uniq_counts[i])
    bodypart_level = [i for i in rem2 if i != individual_level][0]

    scorer = mi.get_level_values(scorer_level).astype(str)
    individual = mi.get_level_values(individual_level).astype(str)
    bodypart = mi.get_level_values(bodypart_level).astype(str)
    coord = mi.get_level_values(coord_level).astype(str)

    out = df.copy()
    out.columns = pd.MultiIndex.from_arrays(
        [scorer, individual, bodypart, coord],
        names=["scorer", "individual", "bodypart", "coord"],
    )
    return out

def _stack_one_camera(
    df: pd.DataFrame,
    camera: str,
    *,
    default_individual: str = "mouse0",
) -> pd.DataFrame:
    """Wide -> long: frame, camera, scorer, individual, bodypart, x, y, likelihood."""
    df = _ensure_expected_multindex(df, default_individual=default_individual)

    long = df.stack(level=[0, 1, 2])  # leaves coord as columns
    long = long.reset_index()

    coord_cols = ["x", "y", "likelihood"]
    for c in coord_cols:
        if c not in long.columns:
            raise RuntimeError(f"Stacked output missing '{c}'. Columns: {list(long.columns)}")

    id_cols = [c for c in long.columns if c not in coord_cols]
    if len(id_cols) != 4:
        id_cols = list(long.columns[:4])

    frame_col, scorer_col, individual_col, bodypart_col = id_cols
    long = long.rename(
        columns={
            frame_col: "frame",
            scorer_col: "scorer",
            individual_col: "individual",
            bodypart_col: "bodypart",
        }
    )
    long.insert(1, "camera", camera)
    long["frame"] = pd.to_numeric(long["frame"], errors="coerce").astype("Int64")

    return long[["frame", "camera", "scorer", "individual", "bodypart", "x", "y", "likelihood"]]


@dataclass
class DLCPredictions:
    long: pd.DataFrame
    cameras: List[str]
    individuals: List[str]
    bodyparts: List[str]
    scorer: str
    frames: int

    def to_array(self) -> np.ndarray:
        """Dense array shape: (T, C, I, B, 3) with (x, y, likelihood)."""
        T = self.frames
        C = len(self.cameras)
        I = len(self.individuals)
        B = len(self.bodyparts)

        arr = np.full((T, C, I, B, 3), np.nan, dtype=np.float32)
        cam_index = {c: i for i, c in enumerate(self.cameras)}
        ind_index = {m: i for i, m in enumerate(self.individuals)}
        bp_index = {b: i for i, b in enumerate(self.bodyparts)}

        df = self.long.dropna(subset=["frame"])
        df = df[df["frame"].between(0, T - 1)]

        for row in df.itertuples(index=False):
            t = int(row.frame)
            c = cam_index[row.camera]
            i = ind_index[row.individual]
            b = bp_index[row.bodypart]
            arr[t, c, i, b, 0] = float(row.x)
            arr[t, c, i, b, 1] = float(row.y)
            arr[t, c, i, b, 2] = float(row.likelihood)

        return arr


def load_dlc_multicam_el(
    camera_to_h5: Dict[str, str | Path],
    *,
    require_same_frames: bool = True,
    default_individual: str = "mouse0",
    pcutoff: Optional[float] = None,
) -> DLCPredictions:
    """Load multiple *_el.h5 files (one per camera) and return normalized predictions."""
    if not camera_to_h5:
        raise ValueError("camera_to_h5 is empty")

    longs: List[pd.DataFrame] = []
    frame_counts: Dict[str, int] = {}
    scorer_names: set[str] = set()

    for cam, path in camera_to_h5.items():
        df = _read_first_dataframe_from_h5(path)
        df = _ensure_expected_multindex(df)

        frame_counts[cam] = len(df.index)
        scorer_names.update(map(str, df.columns.get_level_values(0)))
        longs.append(_stack_one_camera(df, camera=cam))

    long = pd.concat(longs, ignore_index=True)

    if pcutoff is not None:
        long.loc[long["likelihood"] < float(pcutoff), ["x", "y"]] = np.nan

    if len(scorer_names) == 1:
        scorer = next(iter(scorer_names))
    elif len(scorer_names) == 0:
        scorer = ""
    else:
        scorer = sorted(scorer_names)[0]

    cameras = list(camera_to_h5.keys())
    individuals = sorted(long["individual"].dropna().unique().tolist())
    bodyparts = sorted(long["bodypart"].dropna().unique().tolist())

    if require_same_frames:
        counts = set(frame_counts.values())
        if len(counts) != 1:
            raise ValueError(f"Frame counts differ across cameras: {frame_counts}")
        frames = next(iter(counts))
    else:
        frames = max(frame_counts.values())

    long["x"] = pd.to_numeric(long["x"], errors="coerce")
    long["y"] = pd.to_numeric(long["y"], errors="coerce")
    long["likelihood"] = pd.to_numeric(long["likelihood"], errors="coerce")

    return DLCPredictions(
        long=long,
        cameras=cameras,
        individuals=individuals,
        bodyparts=bodyparts,
        scorer=scorer,
        frames=frames,
    )


def _parse_input_specs(specs: List[str]) -> Dict[str, Path]:
    """Parse --input items camname:/path/to/file_or_glob.h5 (globs must match exactly one)."""
    out: Dict[str, Path] = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"--input must be 'camera:/path' but got: {s}")
        cam, p = s.split(":", 1)
        cam = cam.strip()
        p = p.strip()
        if not cam:
            raise ValueError(f"Empty camera name in --input: {s}")

        is_glob = any(ch in p for ch in ["*", "?", "["])
        if is_glob:
            # pathlib.Path.glob() does not support absolute patterns; use glob.glob.
            import glob as _glob
            matches = sorted(Path(m) for m in _glob.glob(os.path.expanduser(p)))
        else:
            matches = [Path(os.path.expanduser(p))]
        matches = [m for m in matches if m.exists()]
        if len(matches) != 1:
            raise ValueError(f"--input {cam}:{p} resolved to {len(matches)} files: {matches}")
        out[cam] = matches[0].resolve()
    return out


def _write_long(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        df.to_csv(out_path, index=False)
    elif suffix == ".parquet":
        try:
            import pyarrow  # noqa: F401
        except Exception as e:
            raise RuntimeError("Parquet output requires 'pyarrow'. Install it or use .csv") from e
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported --out-long extension '{suffix}'. Use .csv or .parquet")


def _write_npz(pred: DLCPredictions, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = pred.to_array()
    np.savez_compressed(
        out_path,
        arr=arr,
        cameras=np.array(pred.cameras, dtype=object),
        individuals=np.array(pred.individuals, dtype=object),
        bodyparts=np.array(pred.bodyparts, dtype=object),
        scorer=np.array(pred.scorer, dtype=object),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Load multi-camera DeepLabCut multi-animal *_el.h5 outputs and export normalized data."
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable. Format: camname:/path/to/file_or_glob_el.h5 (glob must match exactly one file).",
    )
    ap.add_argument(
        "--default-individual",
        default="mouse0",
        help="If the DLC file has 3-level columns (single-animal; no individual level), use this name for the synthetic individual.",
    )
    ap.add_argument(
        "--out-long",
        type=str,
        default="",
        help="Output path for long-format table (.csv or .parquet). If omitted, no table is written.",
    )
    ap.add_argument(
        "--out-npz",
        type=str,
        default="",
        help="Output path for dense array NPZ (arr shape (T,C,I,B,3)). If omitted, no array is written.",
    )
    ap.add_argument(
        "--pcutoff",
        type=float,
        default=None,
        help="If set, rows with likelihood < pcutoff will have x/y set to NaN.",
    )
    ap.add_argument(
        "--allow-different-frames",
        action="store_true",
        help="Allow cameras to have different number of frames; array T becomes max over cameras.",
    )
    args = ap.parse_args()

    camera_to_h5 = _parse_input_specs(args.input)

    pred = load_dlc_multicam_el(
        camera_to_h5,
        require_same_frames=(not args.allow_different_frames),
        default_individual=args.default_individual,
        pcutoff=args.pcutoff,
    )

    print("Loaded DLC predictions:")
    print("  cameras:", pred.cameras)
    print("  individuals:", pred.individuals)
    print("  bodyparts:", pred.bodyparts)
    print("  frames:", pred.frames)
    print("  scorer:", pred.scorer)
    print("  long shape:", pred.long.shape)

    if args.out_long:
        out_path = Path(args.out_long).expanduser().resolve()
        _write_long(pred.long, out_path)
        print("Wrote long table:", out_path)

    if args.out_npz:
        out_path = Path(args.out_npz).expanduser().resolve()
        _write_npz(pred, out_path)
        print("Wrote NPZ array:", out_path)


if __name__ == "__main__":
    main()