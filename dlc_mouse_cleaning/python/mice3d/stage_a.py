from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cameras import CameraRig
from .dlc import Observation
from .geometry import (
    bbox_gate,
    epipolar_distance_px,
    fast_reprojection_errors_px,
    max_angle_pair,
    min_pairwise_ray_angle_rad,
    pair_midpoint_init,
    ray_pair_min_distance,
    build_ray,
    ray_angle_rad,
)


@dataclass
class TriangulationStats:
    rms_px: float
    max_err_px: float
    chi2: float
    residuals_px: np.ndarray  # (N,2)
    num_obs: int
    success: bool
    summary: str


@dataclass
class Hypothesis:
    obs_list: List[Observation]
    kind: str  # 'triplet' or 'pair'
    P_init: np.ndarray
    P_ref: np.ndarray
    stats: TriangulationStats
    score_geom: float


@dataclass
class TriangulatedItem:
    P: Optional[np.ndarray]
    obs_list: List[Observation]
    kind: str  # triplet/pair/single
    geom_ok: bool
    stats: Optional[TriangulationStats] = None

    # Stage B fields
    vote: Tuple[int, int] | None = None
    exclusive: int | None = None
    mouse_id: int | None = None


@dataclass
class StageAParams:
    # gating
    use_epipolar_gate: bool = False
    epi_thresh_px: float = 3.0
    theta_min_triplet_deg: float = 1.5
    theta_min_pair_deg: float = 1.0
    d_max_mm: float = 20.0
    bbox_margin_mm: float = 20.0
    e_init_px: float = 12.0

    # BA accept
    accept_rms_px_triplet: float = 6.0
    accept_rms_px_pair: float = 6.0

    # BA settings
    sigma_pix: float = 2.0
    max_num_iterations: int = 50
    loss: str = "huber"
    loss_scale: float = 1.0
    verbose_ceres: bool = False
    trace_ceres_inputs: bool = False

    # Debugging
    # If True: print every Stage A hypothesis tested and per-gate pass/fail diagnostics.
    debug_stageA: bool = False

    # optional DLC weighting
    use_dlc_weighting: bool = False
    dlc_sigma_min: float = 0.5
    dlc_sigma_max: float = 10.0

    # selection
    triplet_bonus: float = 3.0 #1.5
    # NOTE: This is a *selection* preference (not an acceptance gate).
    # Increase to prefer using 3 cameras (triplets) over 2-camera pairs even when the triplet RMS is modestly higher.
    # Rough rule: set >= (rms_triplet - rms_best_pair) to make that triplet win for the same bodypart.


def _likelihood_to_sigma(lik: float, base_sigma: float, smin: float, smax: float) -> float:
    # lik in [0,1]; higher lik => smaller sigma (stronger weight)
    lik = float(np.clip(lik, 1e-6, 1.0))
    sigma = base_sigma / lik
    return float(np.clip(sigma, smin, smax))


def _solve_point_ba(P_init: np.ndarray,
                    obs_list: List[Observation],
                    rig: CameraRig,
                    params: StageAParams) -> Tuple[np.ndarray, TriangulationStats]:
    try:
        import ceres_point_ba
    except Exception as e:
        raise RuntimeError(
            "ceres_point_ba module not available. Build the extension via ./build.sh"
        ) from e

    cam_idx = np.array([o.cam_idx for o in obs_list], dtype=np.int32)
    uv = np.stack([o.uv for o in obs_list], axis=0).astype(np.float64)

    sigmas = None
    if params.use_dlc_weighting:
        sig = np.array([
            _likelihood_to_sigma(o.likelihood, params.sigma_pix, params.dlc_sigma_min, params.dlc_sigma_max)
            for o in obs_list
        ], dtype=np.float64)
        sigmas = sig


    if params.trace_ceres_inputs:
        # Very verbose: prints every solve input (flush to survive crashes)
        print("[TRACE ceres_point_ba] P_init=", np.asarray(P_init, dtype=np.float64),
              " cam_idx=", cam_idx.tolist(),
              " uv=", uv.tolist(),
              " sigmas=", (None if sigmas is None else sigmas.tolist()),
              flush=True)
    out = ceres_point_ba.solve_point(
        np.asarray(P_init, dtype=np.float64),
        cam_idx,
        uv,
        rig.as_ceres_dict(),
        sigmas,
        float(params.sigma_pix),
        int(params.max_num_iterations),
        str(params.loss),
        float(params.loss_scale),
        bool(params.verbose_ceres),
    )

    P_ref = np.array(out["P"], dtype=float)
    stats = TriangulationStats(
        rms_px=float(out["rms_px"]),
        max_err_px=float(out["max_err_px"]),
        chi2=float(out["chi2"]),
        residuals_px=np.array(out["residuals_px"], dtype=float),
        num_obs=int(out["num_obs"]),
        success=bool(out["success"]),
        summary=str(out["summary"]),
    )
    return P_ref, stats


def _geom_score(stats: TriangulationStats, kind: str, params: StageAParams) -> float:
    """Return a *higher-is-better* score for hypothesis selection.

    Stage A selection can consider up to 2 disjoint hypotheses (two mice). If we
    use a negative quantity like -chi2, then selecting 2 hypotheses always makes
    the summed score *worse*, and comparing against a "none" option with score 0
    can cause everything to be dropped.

    We therefore score by the *margin to the acceptance threshold* (in pixels),
    which is naturally positive for accepted hypotheses and additive across
    multiple animals.
    """
    thr = float(params.accept_rms_px_triplet if kind == "triplet" else params.accept_rms_px_pair)
    score = thr - float(stats.rms_px)

    # Tiny epsilon so an exactly-on-threshold solution doesn't tie with "none".
    score += 1e-6

    if kind == "triplet":
        score += float(params.triplet_bonus)
    return float(score)


def generate_triplet_candidates(obs_all: List[Observation], cam_count: int = 3, ids: Sequence[int] = (0, 1)) -> List[List[Observation]]:
    # Expect cams 0..cam_count-1
    obs_by_cam_id: Dict[Tuple[int, int], Observation] = {(o.cam_idx, o.dlc_id): o for o in obs_all}
    out: List[List[Observation]] = []
    # assumes exactly 3 cameras (0,1,2). Extendable but keep simple.
    cams = list(range(cam_count))
    if len(cams) != 3:
        raise ValueError("generate_triplet_candidates currently assumes 3 cameras")

    for i in ids:
        for j in ids:
            for k in ids:
                a = obs_by_cam_id.get((cams[0], i))
                b = obs_by_cam_id.get((cams[1], j))
                c = obs_by_cam_id.get((cams[2], k))
                if a is None or b is None or c is None:
                    continue
                out.append([a, b, c])
    return out


def generate_pair_candidates(obs_all: List[Observation], cam_pairs: Sequence[Tuple[int, int]] = ((0, 1), (0, 2), (1, 2)), ids: Sequence[int] = (0, 1)) -> List[List[Observation]]:
    obs_by_cam_id: Dict[Tuple[int, int], Observation] = {(o.cam_idx, o.dlc_id): o for o in obs_all}
    out: List[List[Observation]] = []

    for (c1, c2) in cam_pairs:
        for i in ids:
            for j in ids:
                a = obs_by_cam_id.get((c1, i))
                b = obs_by_cam_id.get((c2, j))
                if a is None or b is None:
                    continue
                out.append([a, b])
    return out


def evaluate_hypotheses(
    rig: CameraRig,
    triplets: List[List[Observation]],
    pairs: List[List[Observation]],
    params: StageAParams,
) -> List[Hypothesis]:
    """Evaluate all candidate triplet/pair hypotheses for a single bodypart.

    When params.debug_stageA is enabled, prints:
      - the hypothesis (bodypart + per-camera observations),
      - each pre-BA gate (PASS/FAIL with key values),
      - BA outcome (Ceres success + summary),
      - final ACCEPT/REJECT decision.
    """
    valid: List[Hypothesis] = []

    theta_trip = np.deg2rad(params.theta_min_triplet_deg)
    theta_pair = np.deg2rad(params.theta_min_pair_deg)

    # Precompute fundamental matrices if needed
    F_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _ctx(obs_list: List[Observation], kind: str) -> Tuple[int, str, str]:
        frame = int(obs_list[0].frame) if obs_list else -1
        bp = str(obs_list[0].bodypart) if obs_list else "?"
        return frame, bp, kind

    def _dbg(obs_list: List[Observation], kind: str, msg: str) -> None:
        if not params.debug_stageA:
            return
        frame, bp, k = _ctx(obs_list, kind)
        print(f"[DEBUG stageA] frame={frame} bodypart={bp} kind={k} :: {msg}", flush=True)

    def _fmt_obs(o: Observation) -> str:
        u, v = float(o.uv[0]), float(o.uv[1])
        return (
            f"{o.cam_name}(cam_idx={int(o.cam_idx)},dlc_id={int(o.dlc_id)})"
            f" uv=({u:.3f},{v:.3f}) lik={float(o.likelihood):.3f}"
        )

    def _print_hypothesis(kind: str, obs_list: List[Observation]) -> None:
        if not params.debug_stageA:
            return
        _dbg(obs_list, kind, "TEST :: " + " | ".join(_fmt_obs(o) for o in obs_list))

    def epi_dist(a: Observation, b: Observation) -> float:
        # Distance of b to epipolar line induced by a
        if not params.use_epipolar_gate:
            return 0.0
        key = (a.cam_idx, b.cam_idx)
        if key not in F_cache:
            F_cache[key] = None  # computed inside epipolar_distance_px
        return float(epipolar_distance_px(rig, a, b, F_cache[key]))

    def epi_ok(a: Observation, b: Observation) -> bool:
        if not params.use_epipolar_gate:
            return True
        return epi_dist(a, b) <= params.epi_thresh_px

    # -------------------------
    # Triplets
    # -------------------------
    for obs_list in triplets:
        kind = "triplet"
        _print_hypothesis(kind, obs_list)

        # Gate 0: epipolar
        if params.use_epipolar_gate:
            d01 = epi_dist(obs_list[0], obs_list[1])
            ok01 = d01 <= params.epi_thresh_px
            _dbg(obs_list, kind, f"GATE epipolar 0->1 d={d01:.3f}px thr={params.epi_thresh_px:.3f}px -> {'PASS' if ok01 else 'FAIL'}")
            if not ok01:
                _dbg(obs_list, kind, "RESULT REJECTED reason=epipolar_0_1")
                continue

            d02 = epi_dist(obs_list[0], obs_list[2])
            ok02 = d02 <= params.epi_thresh_px
            _dbg(obs_list, kind, f"GATE epipolar 0->2 d={d02:.3f}px thr={params.epi_thresh_px:.3f}px -> {'PASS' if ok02 else 'FAIL'}")
            if not ok02:
                _dbg(obs_list, kind, "RESULT REJECTED reason=epipolar_0_2")
                continue

            d12 = epi_dist(obs_list[1], obs_list[2])
            ok12 = d12 <= params.epi_thresh_px
            _dbg(obs_list, kind, f"GATE epipolar 1->2 d={d12:.3f}px thr={params.epi_thresh_px:.3f}px -> {'PASS' if ok12 else 'FAIL'}")
            if not ok12:
                _dbg(obs_list, kind, "RESULT REJECTED reason=epipolar_1_2")
                continue

        # Gate 1: ray angle (min pairwise)
        ang_min = float(min_pairwise_ray_angle_rad(rig, obs_list))
        ang_min_deg = float(np.rad2deg(ang_min))
        ok_ang = ang_min >= theta_trip
        _dbg(obs_list, kind, f"GATE ray_angle min={ang_min_deg:.3f}deg thr={params.theta_min_triplet_deg:.3f}deg -> {'PASS' if ok_ang else 'FAIL'}")
        if not ok_ang:
            _dbg(obs_list, kind, "RESULT REJECTED reason=ray_angle")
            continue

        # Gate 2: closest approach distances (pairwise)
        pair_dists = []
        ok_dist = True
        fail_pair = None
        for ii in range(3):
            for jj in range(ii + 1, 3):
                r1 = build_ray(rig, obs_list[ii])
                r2 = build_ray(rig, obs_list[jj])
                dmm = float(ray_pair_min_distance(r1, r2))
                pair_dists.append(((ii, jj), dmm))
                if dmm > params.d_max_mm and ok_dist:
                    ok_dist = False
                    fail_pair = (ii, jj, dmm)
        if ok_dist:
            max_d = max(d for (_, d) in pair_dists) if pair_dists else float("nan")
            _dbg(obs_list, kind, f"GATE ray_closest_approach max={max_d:.3f}mm thr={params.d_max_mm:.3f}mm -> PASS")
        else:
            ii, jj, dmm = fail_pair
            _dbg(obs_list, kind, f"GATE ray_closest_approach pair=({ii},{jj}) d={dmm:.3f}mm thr={params.d_max_mm:.3f}mm -> FAIL")
            _dbg(obs_list, kind, "RESULT REJECTED reason=ray_closest_approach")
            continue

        # Init from best-conditioned pair
        bi, bj = max_angle_pair(rig, obs_list)
        P_init = pair_midpoint_init(rig, obs_list[bi], obs_list[bj])
        _dbg(obs_list, kind, f"INIT best_pair=({bi},{bj}) P_init=({float(P_init[0]):.3f},{float(P_init[1]):.3f},{float(P_init[2]):.3f})")

        # Gate 3: bbox
        ok_bbox = bool(bbox_gate(P_init, rig, params.bbox_margin_mm))
        _dbg(obs_list, kind, f"GATE bbox -> {'PASS' if ok_bbox else 'FAIL'} (margin_mm={params.bbox_margin_mm:.3f})")
        if not ok_bbox:
            _dbg(obs_list, kind, "RESULT REJECTED reason=bbox")
            continue

        # Gate 4: fast reproj
        errs = fast_reprojection_errors_px(rig, P_init, obs_list)
        max_e = float(np.max(errs)) if errs.size else float("inf")
        ok_reproj = max_e <= params.e_init_px
        if params.debug_stageA:
            err_parts = []
            for o, e in zip(obs_list, errs.tolist()):
                err_parts.append(f"{o.cam_name}:{float(e):.3f}px")
            _dbg(obs_list, kind, f"GATE fast_reproj max={max_e:.3f}px thr={params.e_init_px:.3f}px -> {'PASS' if ok_reproj else 'FAIL'} ({', '.join(err_parts)})")
        if not ok_reproj:
            _dbg(obs_list, kind, "RESULT REJECTED reason=fast_reproj")
            continue

        # BA
        P_ref, stats = _solve_point_ba(P_init, obs_list, rig, params)
        _dbg(
            obs_list,
            kind,
            "BA "
            + f"success={bool(stats.success)} rms_px={float(stats.rms_px):.3f} "
            + f"max_err_px={float(stats.max_err_px):.3f} chi2={float(stats.chi2):.3f} "
            + f"thr_rms={params.accept_rms_px_triplet:.3f} summary='{stats.summary}'",
        )

        if not stats.success:
            _dbg(obs_list, kind, "RESULT REJECTED reason=ceres_not_usable")
            continue
        if stats.rms_px > params.accept_rms_px_triplet:
            _dbg(obs_list, kind, "RESULT REJECTED reason=rms_px_threshold")
            continue

        _dbg(obs_list, kind, "RESULT ACCEPTED")
        valid.append(
            Hypothesis(
                obs_list=obs_list,
                kind="triplet",
                P_init=P_init,
                P_ref=P_ref,
                stats=stats,
                score_geom=_geom_score(stats, "triplet", params),
            )
        )

    # -------------------------
    # Pairs
    # -------------------------
    for obs_list in pairs:
        kind = "pair"
        _print_hypothesis(kind, obs_list)

        # Gate 0: epipolar
        if params.use_epipolar_gate:
            d01 = epi_dist(obs_list[0], obs_list[1])
            ok01 = d01 <= params.epi_thresh_px
            _dbg(obs_list, kind, f"GATE epipolar 0->1 d={d01:.3f}px thr={params.epi_thresh_px:.3f}px -> {'PASS' if ok01 else 'FAIL'}")
            if not ok01:
                _dbg(obs_list, kind, "RESULT REJECTED reason=epipolar_0_1")
                continue

        # Gate 1: ray angle
        r1 = build_ray(rig, obs_list[0])
        r2 = build_ray(rig, obs_list[1])
        ang = float(ray_angle_rad(r1, r2))
        ang_deg = float(np.rad2deg(ang))
        ok_ang = ang >= theta_pair
        _dbg(obs_list, kind, f"GATE ray_angle angle={ang_deg:.3f}deg thr={params.theta_min_pair_deg:.3f}deg -> {'PASS' if ok_ang else 'FAIL'}")
        if not ok_ang:
            _dbg(obs_list, kind, "RESULT REJECTED reason=ray_angle")
            continue

        # Gate 2: closest approach
        dmm = float(ray_pair_min_distance(r1, r2))
        ok_dist = dmm <= params.d_max_mm
        _dbg(obs_list, kind, f"GATE ray_closest_approach d={dmm:.3f}mm thr={params.d_max_mm:.3f}mm -> {'PASS' if ok_dist else 'FAIL'}")
        if not ok_dist:
            _dbg(obs_list, kind, "RESULT REJECTED reason=ray_closest_approach")
            continue

        # Init
        P_init = pair_midpoint_init(rig, obs_list[0], obs_list[1])
        _dbg(obs_list, kind, f"INIT P_init=({float(P_init[0]):.3f},{float(P_init[1]):.3f},{float(P_init[2]):.3f})")

        # Gate 3: bbox
        ok_bbox = bool(bbox_gate(P_init, rig, params.bbox_margin_mm))
        _dbg(obs_list, kind, f"GATE bbox -> {'PASS' if ok_bbox else 'FAIL'} (margin_mm={params.bbox_margin_mm:.3f})")
        if not ok_bbox:
            _dbg(obs_list, kind, "RESULT REJECTED reason=bbox")
            continue

        # Gate 4: fast reproj
        errs = fast_reprojection_errors_px(rig, P_init, obs_list)
        max_e = float(np.max(errs)) if errs.size else float("inf")
        ok_reproj = max_e <= params.e_init_px
        if params.debug_stageA:
            err_parts = []
            for o, e in zip(obs_list, errs.tolist()):
                err_parts.append(f"{o.cam_name}:{float(e):.3f}px")
            _dbg(obs_list, kind, f"GATE fast_reproj max={max_e:.3f}px thr={params.e_init_px:.3f}px -> {'PASS' if ok_reproj else 'FAIL'} ({', '.join(err_parts)})")
        if not ok_reproj:
            _dbg(obs_list, kind, "RESULT REJECTED reason=fast_reproj")
            continue

        # BA
        P_ref, stats = _solve_point_ba(P_init, obs_list, rig, params)
        _dbg(
            obs_list,
            kind,
            "BA "
            + f"success={bool(stats.success)} rms_px={float(stats.rms_px):.3f} "
            + f"max_err_px={float(stats.max_err_px):.3f} chi2={float(stats.chi2):.3f} "
            + f"thr_rms={params.accept_rms_px_pair:.3f} summary='{stats.summary}'",
        )

        if not stats.success:
            _dbg(obs_list, kind, "RESULT REJECTED reason=ceres_not_usable")
            continue
        if stats.rms_px > params.accept_rms_px_pair:
            _dbg(obs_list, kind, "RESULT REJECTED reason=rms_px_threshold")
            continue

        _dbg(obs_list, kind, "RESULT ACCEPTED")
        valid.append(
            Hypothesis(
                obs_list=obs_list,
                kind="pair",
                P_init=P_init,
                P_ref=P_ref,
                stats=stats,
                score_geom=_geom_score(stats, "pair", params),
            )
        )

    return valid


def select_best_disjoint(hypotheses: List[Hypothesis], max_points: int = 2) -> List[Hypothesis]:
    # Brute-force search over choosing up to 2 hypotheses.
    best: List[Hypothesis] = []
    best_score = -1e99

    # Include option of selecting none
    H = [None] + hypotheses

    def disjoint(sel: List[Hypothesis]) -> bool:
        used = set()
        for h in sel:
            for o in h.obs_list:
                key = (o.cam_idx, o.dlc_id)
                if key in used:
                    return False
                used.add(key)
        return True

    for h1 in H:
        for h2 in H:
            sel = []
            if h1 is not None:
                sel.append(h1)
            if h2 is not None and h2 is not h1:
                sel.append(h2)
            if len(sel) > max_points:
                continue
            if not disjoint(sel):
                continue
            score = (sum(h.score_geom for h in sel) if sel else -1e99)
            if score > best_score:
                best_score = score
                best = sel

    # sort best by score descending for deterministic output
    return sorted(best, key=lambda h: h.score_geom, reverse=True)


def stage_a_for_bodypart(rig: CameraRig, obs_all: List[Observation], params: StageAParams) -> List[TriangulatedItem]:
    if not obs_all:
        return []

    triplets = generate_triplet_candidates(obs_all, cam_count=len(rig.cameras), ids=(0, 1))
    pairs = generate_pair_candidates(obs_all, cam_pairs=((0, 1), (0, 2), (1, 2)), ids=(0, 1))

    valid = evaluate_hypotheses(rig, triplets, pairs, params)
    chosen = select_best_disjoint(valid, max_points=2)

    used_obs = set()
    items: List[TriangulatedItem] = []

    for h in chosen:
        for o in h.obs_list:
            used_obs.add((o.cam_idx, o.dlc_id))
        items.append(
            TriangulatedItem(
                P=h.P_ref,
                obs_list=h.obs_list,
                kind=h.kind,
                geom_ok=True,
                stats=h.stats,
            )
        )

    # leftover singles
    for o in obs_all:
        if (o.cam_idx, o.dlc_id) not in used_obs:
            items.append(TriangulatedItem(P=None, obs_list=[o], kind="single", geom_ok=False))

        # Final per-bodypart selection summary (debug)
    if params.debug_stageA:
        frame = int(obs_all[0].frame)
        bp = str(obs_all[0].bodypart)
        valid_n = len(valid)
        chosen_n = len(chosen)
        best_single = float("nan")
        if valid:
            best_single = max((h.score_geom for h in valid))
        score_sum = float(sum(h.score_geom for h in chosen))
        note = ""
        # select_best_disjoint currently allows selecting "none" (score 0). If all hypotheses have negative scores,
        # this can intentionally choose none; this note helps diagnose that case.
        if chosen_n == 0 and valid_n > 0:
            note = " NOTE unexpected: valid_hypotheses_but_none_selected"
        print(
            f"[DEBUG stageA] frame={frame} bodypart={bp} kind=final :: "
            f"SELECT chosen={chosen_n} valid={valid_n} score_sum={score_sum:.3f} best_single={best_single:.3f}{note}",
            flush=True,
        )

        def _term(summary: str) -> str:
            if "Termination:" not in summary:
                return ""
            # e.g. "... Termination: CONVERGENCE" or "Termination: NO_CONVERGENCE"
            tail = summary.split("Termination:", 1)[1].strip()
            # trim after comma if present
            if "," in tail:
                tail = tail.split(",", 1)[0].strip()
            # first token
            return tail.split()[0].strip()

        for i, h in enumerate(chosen):
            cams = [int(o.cam_idx) for o in h.obs_list]
            dlc_ids = [int(o.dlc_id) for o in h.obs_list]
            uv = [(float(o.uv[0]), float(o.uv[1])) for o in h.obs_list]
            P = h.P_ref
            term = _term(h.stats.summary)
            print(
                f"[DEBUG stageA] frame={frame} bodypart={bp} kind=final :: "
                f"CHOSEN#{i} kind={h.kind} score={h.score_geom:.3f} "
                f"P=({float(P[0]):.3f},{float(P[1]):.3f},{float(P[2]):.3f}) "
                f"rms_px={h.stats.rms_px:.3f} max_err_px={h.stats.max_err_px:.3f} chi2={h.stats.chi2:.3f} "
                f"success={bool(h.stats.success)} term={term} cams={cams} dlc_ids={dlc_ids} uv={uv}",
                flush=True,
            )

        singles = [o for o in obs_all if (o.cam_idx, o.dlc_id) not in used_obs]
        if singles:
            ss = " | ".join([
                f"{o.cam_name}(cam_idx={int(o.cam_idx)},dlc_id={int(o.dlc_id)}) "
                f"uv=({float(o.uv[0]):.3f},{float(o.uv[1]):.3f}) lik={float(o.likelihood):.3f}"
                for o in singles
            ])
            print(f"[DEBUG stageA] frame={frame} bodypart={bp} kind=final :: SINGLES {len(singles)} :: {ss}", flush=True)
        else:
            print(f"[DEBUG stageA] frame={frame} bodypart={bp} kind=final :: SINGLES 0", flush=True)

    return items
