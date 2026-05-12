from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DLCIndex:
    camera_names: List[str]
    bodyparts: List[str]
    individuals: List[str]
    cam_to_idx: Dict[str, int]
    bp_to_idx: Dict[str, int]
    ind_to_idx: Dict[str, int]


@dataclass
class DLCData:
    """Dense DLC storage: arrays [T,C,I,B]."""

    index: DLCIndex
    x: np.ndarray  # (T,C,I,B)
    y: np.ndarray  # (T,C,I,B)
    l: np.ndarray  # (T,C,I,B)

    @property
    def num_frames(self) -> int:
        return self.x.shape[0]


@dataclass
class Observation:
    frame: int
    cam_idx: int
    cam_name: str
    dlc_id: int  # 0 or 1
    bodypart_idx: int
    bodypart: str
    uv: np.ndarray  # (2,)
    likelihood: float

    # Filled later
    seg_w: Tuple[float, float, float] | None = None
    vote: Tuple[int, int] | None = None


def load_dlc_long_csv(path: str, camera_order: List[str] | None = None) -> DLCData:
    df = pd.read_csv(path)

    # Determine index sets
    camera_names = list(df["camera"].unique())
    if camera_order is not None:
        # Keep only cameras present in csv
        camera_names = [c for c in camera_order if c in set(camera_names)]
    else:
        camera_names = sorted(camera_names)

    bodyparts = sorted(df["bodypart"].unique())
    individuals = list(df["individual"].unique())
    # enforce stable order mouse0, mouse1 when present
    if set(individuals) >= {"mouse0", "mouse1"}:
        individuals = ["mouse0", "mouse1"]
    else:
        individuals = sorted(individuals)

    cam_to_idx = {c: i for i, c in enumerate(camera_names)}
    bp_to_idx = {b: i for i, b in enumerate(bodyparts)}
    ind_to_idx = {ind: i for i, ind in enumerate(individuals)}

    T = int(df["frame"].max()) + 1
    C = len(camera_names)
    I = len(individuals)
    B = len(bodyparts)

    x = np.full((T, C, I, B), np.nan, dtype=float)
    y = np.full((T, C, I, B), np.nan, dtype=float)
    l = np.zeros((T, C, I, B), dtype=float)

    for row in df.itertuples(index=False):
        f = int(row.frame)
        cam = str(row.camera)
        ind = str(row.individual)
        bp = str(row.bodypart)
        if cam not in cam_to_idx or ind not in ind_to_idx or bp not in bp_to_idx:
            continue
        ci = cam_to_idx[cam]
        ii = ind_to_idx[ind]
        bi = bp_to_idx[bp]
        x[f, ci, ii, bi] = float(row.x) if pd.notna(row.x) else np.nan
        y[f, ci, ii, bi] = float(row.y) if pd.notna(row.y) else np.nan
        l[f, ci, ii, bi] = float(row.likelihood) if pd.notna(row.likelihood) else 0.0

    index = DLCIndex(
        camera_names=camera_names,
        bodyparts=bodyparts,
        individuals=individuals,
        cam_to_idx=cam_to_idx,
        bp_to_idx=bp_to_idx,
        ind_to_idx=ind_to_idx,
    )

    return DLCData(index=index, x=x, y=y, l=l)


def build_observations(dlc: DLCData, frame: int, bodypart_idx: int, likelihood_min: float = 0.0) -> List[Observation]:
    idx = dlc.index
    obs: List[Observation] = []

    for cam_idx, cam_name in enumerate(idx.camera_names):
        for dlc_id in range(len(idx.individuals)):
            u = dlc.x[frame, cam_idx, dlc_id, bodypart_idx]
            v = dlc.y[frame, cam_idx, dlc_id, bodypart_idx]
            lik = dlc.l[frame, cam_idx, dlc_id, bodypart_idx]
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            if lik < likelihood_min:
                continue
            obs.append(
                Observation(
                    frame=frame,
                    cam_idx=cam_idx,
                    cam_name=cam_name,
                    dlc_id=dlc_id,
                    bodypart_idx=bodypart_idx,
                    bodypart=idx.bodyparts[bodypart_idx],
                    uv=np.array([u, v], dtype=float),
                    likelihood=float(lik),
                )
            )

    return obs
