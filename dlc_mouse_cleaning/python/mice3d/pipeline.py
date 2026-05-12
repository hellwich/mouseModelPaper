from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cameras import CameraRig
from .dlc import DLCData, Observation, build_observations
from .segmentation import SegmentationSource
from .stage_a import StageAParams, TriangulatedItem, stage_a_for_bodypart
from .stage_b import PrevState, StageBParams, stage_b_assign


@dataclass
class ReconstructionParams:
    likelihood_min: float = 0.0
    # Use default_factory to avoid shared mutable defaults.
    stage_a: StageAParams = field(default_factory=StageAParams)
    stage_b: StageBParams = field(default_factory=StageBParams)


class Mouse3DReconstructor:
    def __init__(
        self,
        rig: CameraRig,
        dlc: DLCData,
        seg_source: Optional[SegmentationSource] = None,
        params: ReconstructionParams = ReconstructionParams(),
        debug: bool = False,
        debug_save_path: Optional[str] = None,
    ) -> None:
        self.rig = rig
        self.dlc = dlc
        self.seg_source = seg_source
        self.params = params

        self.prev_state = PrevState()

        self.debug = bool(debug)
        self.debug_save_path = debug_save_path

        # outputs
        self._triang_rows: List[dict] = []
        self._assign_rows: List[dict] = []

        # create bodypart list
        self.bodyparts = dlc.index.bodyparts

    def process_frame(self, frame: int) -> Dict[str, List[TriangulatedItem]]:
        # Stage A
        stageA_results: Dict[str, List[TriangulatedItem]] = {}
        for bi, b in enumerate(self.bodyparts):
            obs_all: List[Observation] = build_observations(self.dlc, frame, bi, likelihood_min=self.params.likelihood_min)
            items = stage_a_for_bodypart(self.rig, obs_all, self.params.stage_a)
            stageA_results[b] = items

        # Stage B
        stageB_results, new_state = stage_b_assign(
            frame,
            self.rig,
            stageA_results,
            self.seg_source,
            self.prev_state,
            self.params.stage_b,
        )
        self.prev_state = new_state

        # Record outputs
        self._record_frame(frame, stageB_results)
        self._debug_frame(frame, stageA_results, stageB_results)
        return stageB_results

    def run_all(self, frames: Optional[List[int]] = None) -> None:
        if frames is None:
            frames = list(range(self.dlc.num_frames))
        for f in frames:
            self.process_frame(f)

    def _record_frame(self, frame: int, results: Dict[str, List[TriangulatedItem]]) -> None:
        # 3D outputs
        for b, items in results.items():
            for item in items:
                if item.P is None:
                    continue
                if item.mouse_id is None:
                    continue
                row = {
                    "frame": frame,
                    "bodypart": b,
                    "mouse_id": int(item.mouse_id),
                    "X": float(item.P[0]),
                    "Y": float(item.P[1]),
                    "Z": float(item.P[2]),
                    "kind": item.kind,
                    "num_obs": int(item.stats.num_obs) if item.stats else len(item.obs_list),
                    "rms_px": float(item.stats.rms_px) if item.stats else np.nan,
                    "max_err_px": float(item.stats.max_err_px) if item.stats else np.nan,
                    "chi2": float(item.stats.chi2) if item.stats else np.nan,
                }
                self._triang_rows.append(row)

        # assignment outputs per original DLC observation
        for b, items in results.items():
            for item in items:
                for obs in item.obs_list:
                    self._assign_rows.append(
                        {
                            "frame": frame,
                            "camera": obs.cam_name,
                            "bodypart": b,
                            "dlc_individual": obs.dlc_id,
                            "x": float(obs.uv[0]),
                            "y": float(obs.uv[1]),
                            "likelihood": float(obs.likelihood),
                            "new_mouse_id": None if item.mouse_id is None else int(item.mouse_id),
                            "seg_w_bg": None if obs.seg_w is None else float(obs.seg_w[0]),
                            "seg_w_m0": None if obs.seg_w is None else float(obs.seg_w[1]),
                            "seg_w_m1": None if obs.seg_w is None else float(obs.seg_w[2]),
                            "vote0": None if obs.vote is None else int(obs.vote[0]),
                            "vote1": None if obs.vote is None else int(obs.vote[1]),
                            "item_kind": item.kind,
                        }
                    )



    def _debug_frame(self, frame: int, stageA: Dict[str, List[TriangulatedItem]], stageB: Dict[str, List[TriangulatedItem]]) -> None:
        """Optional per-frame debugging.

        - If self.debug: prints a compact per-frame summary.
        - If self.debug_save_path: appends a JSON line with per-bodypart details.
        """
        if not (self.debug or self.debug_save_path):
            return

        if self.debug:
            n3d = 0
            nitems = 0
            rms_vals = []
            for _, items in stageB.items():
                for it in items:
                    nitems += 1
                    if it.P is not None:
                        n3d += 1
                        if it.stats is not None and np.isfinite(it.stats.rms_px):
                            rms_vals.append(float(it.stats.rms_px))
            rms_mean = float(np.mean(rms_vals)) if rms_vals else float('nan')
            print(f"[frame {frame}] items={nitems} 3D={n3d} mean_rms_px={rms_mean:.3f} swap={self.prev_state.swap_state}")

        if self.debug_save_path:
            rec = {"frame": int(frame), "swap_state": str(self.prev_state.swap_state), "bodyparts": {}}
            for b, items in stageB.items():
                out_items = []
                for it in items:
                    out_items.append({
                        "kind": it.kind,
                        "geom_ok": bool(it.geom_ok),
                        "mouse_id": None if it.mouse_id is None else int(it.mouse_id),
                        "P": None if it.P is None else [float(it.P[0]), float(it.P[1]), float(it.P[2])],
                        "stats": None if it.stats is None else {
                            "rms_px": float(it.stats.rms_px),
                            "max_err_px": float(it.stats.max_err_px),
                            "chi2": float(it.stats.chi2),
                            "num_obs": int(it.stats.num_obs),
                            "success": bool(it.stats.success),
                        },
                        "vote": None if it.vote is None else [int(it.vote[0]), int(it.vote[1])],
                        "exclusive": it.exclusive,
                        "obs": [
                            {
                                "cam": o.cam_name,
                                "cam_idx": int(o.cam_idx),
                                "dlc_id": int(o.dlc_id),
                                "uv": [float(o.uv[0]), float(o.uv[1])],
                                "lik": float(o.likelihood),
                                "seg_w": None if o.seg_w is None else [float(o.seg_w[0]), float(o.seg_w[1]), float(o.seg_w[2])],
                                "vote": None if o.vote is None else [int(o.vote[0]), int(o.vote[1])],
                            }
                            for o in it.obs_list
                        ],
                    })
                rec["bodyparts"][b] = out_items

            with open(self.debug_save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def triangulated_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._triang_rows)

    def assignments_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._assign_rows)

    def save_outputs(self, out_prefix: str) -> Tuple[str, str]:
        tri_path = out_prefix + "_triangulated3d.csv"
        ass_path = out_prefix + "_dlc_assignments.csv"
        from pathlib import Path
        Path(tri_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ass_path).parent.mkdir(parents=True, exist_ok=True)
        self.triangulated_dataframe().to_csv(tri_path, index=False)
        self.assignments_dataframe().to_csv(ass_path, index=False)
        return tri_path, ass_path
