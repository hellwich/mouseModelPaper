from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

from functools import lru_cache

import numpy as np

from .cameras import CameraRig
from .dlc import Observation
from .segmentation import GaussianKernel, SegmentationSource, seg_weights_from_mask
from .geometry import distance_point_to_line
from .stage_a import TriangulatedItem


def _coalesce(a, b):
    return a if a is not None else b


class SwapState:
    NOSWAP = "NOSWAP"
    SWAP = "SWAP"
    UNKNOWN = "UNKNOWN"


# -------------------------
# Graph model (B7)
# -------------------------


@dataclass
class GraphModel:
    nodes: Dict[str, np.ndarray]  # name -> (3,)
    edges: List[Tuple[str, str]]
    neighbors: Dict[str, List[str]]
    edge_len: Dict[Tuple[str, str], float]  # sorted endpoints -> length

    def is_leaf(self, name: str) -> bool:
        return len(self.neighbors.get(name, [])) == 1

    def leaf_nodes(self) -> List[str]:
        return [n for n in self.nodes.keys() if self.is_leaf(n)]


def load_graph_model(path: str) -> GraphModel:
    """Load a graph model from the provided text format.

    Expected format (example):

      [NODES]
      head  26 0 28
      ...

      [EDGES]
      head nose_tip
      ...

    Coordinates are assumed to be in mm.
    """
    nodes: Dict[str, np.ndarray] = {}
    edges: List[Tuple[str, str]] = []

    mode = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.upper() == "[NODES]":
                mode = "nodes"
                continue
            if line.upper() == "[EDGES]":
                mode = "edges"
                continue

            if mode == "nodes":
                parts = line.split()
                if len(parts) < 4:
                    continue
                name = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except Exception:
                    continue
                nodes[name] = np.array([x, y, z], dtype=float)
            elif mode == "edges":
                parts = line.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
                edges.append((a, b))

    neighbors: Dict[str, List[str]] = {k: [] for k in nodes.keys()}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        if a not in nodes or b not in nodes:
            # ignore dangling edge
            continue
        neighbors[a].append(b)
        neighbors[b].append(a)
        key = tuple(sorted((a, b)))
        edge_len[key] = float(np.linalg.norm(nodes[a] - nodes[b]))

    return GraphModel(nodes=nodes, edges=[(a, b) for (a, b) in edges if a in nodes and b in nodes], neighbors=neighbors, edge_len=edge_len)


@lru_cache(maxsize=8)
def _load_graph_model_cached(path: str) -> GraphModel:
    """Cached graph loader to avoid re-parsing on every frame."""
    return load_graph_model(path)


def _kabsch_rigid(model_pts: np.ndarray, world_pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Estimate rigid transform mapping model_pts -> world_pts.

    Returns (R, t, rms) where world ≈ R @ model + t.
    """
    if model_pts.shape[0] < 3:
        return None
    if model_pts.shape != world_pts.shape:
        return None

    X = np.asarray(model_pts, dtype=float)
    Y = np.asarray(world_pts, dtype=float)

    cx = X.mean(axis=0)
    cy = Y.mean(axis=0)
    Xc = X - cx
    Yc = Y - cy

    H = Xc.T @ Yc
    try:
        U, S, Vt = np.linalg.svd(H)
    except Exception:
        return None

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cy - R @ cx

    Yhat = (X @ R.T) + t  # X row-vectors; equivalent to (R@x + t)
    rms = float(np.sqrt(np.mean(np.sum((Yhat - Y) ** 2, axis=1))))
    return R, t, rms


# -------------------------
# Stage B params/state
# -------------------------


@dataclass
class StageBParams:
    # segmentation vote
    vote_thr: float = 0.01
    seg_radius: int = 9
    seg_sigma: float = 4.0

    # swap continuity
    tau_swap: float = 4.0
    carry_swap_max_gap: int = 5
    sigma_motion_anchor: float = 8.0  # mm

    # weak exclusive bias in swap decision
    penalty_excl_swap: float = 1.0
    bonus_excl_swap: float = 0.25

    # identity assignment costs
    sigma_motion_other: float = 15.0
    sigma_centroid: float = 60.0
    sigma_axis: float = 35.0
    w_cont: float = 1.0
    w_cent: float = 1.0
    w_axis: float = 1.0

    # exclusive strength during final identity assignment
    hard_exclusive: bool = False
    penalty_excl: float = 9.0
    bonus_excl: float = 0.5

    # anchors
    anchor_parts: Tuple[str, str] = ("head", "tail_root")

    # continuity fallback for singles
    single_proj_margin_px: float = 0.0  # if >0 require a margin to decide

    # Debugging
    debug_stageB: bool = False

    # B7 graph plausibility (optional; enabled when graph_model is not None)
    graph_model: Optional[str] = None
    graph_min_ratio: float = 2.0 / 3.0
    graph_max_ratio: float = 1.5
    graph_leaf_only: bool = True

    # Rigid fallback for disconnected leaf nodes (mm)
    graph_rigid_tol_ratio: float = 1.0
    # Max absolute allowed deviation between a leaf's estimated 3D position and the
    # rigidly-transformed template position (Kabsch fit). In this simulation setup,
    # a tight tolerance is appropriate; otherwise rigid fallback can override an
    # edge-length violation with a still-large positional deviation.
    graph_rigid_tol_mm_max: float = 5.0

    # If True, drop leaf points when undecidable (no neighbor + no rigid fit)
    graph_drop_if_undecidable: bool = False


@dataclass
class PrevState:
    # previous 3D positions per bodypart and mouse
    prev: Dict[str, Dict[int, Optional[np.ndarray]]] = field(default_factory=dict)
    prev2: Dict[str, Dict[int, Optional[np.ndarray]]] = field(default_factory=dict)

    # anchor-specific (alias, but kept separate for readability)
    anchor_prev: Dict[str, Dict[int, Optional[np.ndarray]]] = field(default_factory=dict)
    anchor_prev2: Dict[str, Dict[int, Optional[np.ndarray]]] = field(default_factory=dict)

    # last centroid/axis models
    centroid_prev: Dict[int, Optional[np.ndarray]] = field(default_factory=lambda: {0: None, 1: None})
    axis_prev: Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=lambda: {0: None, 1: None})

    swap_state: str = SwapState.UNKNOWN
    swap_age: int = 999999


# -------------------------
# Debug helpers
# -------------------------


_DEBUG_STAGEB_FRAME: Optional[int] = None


def _dbg_stageB(params: StageBParams, msg: str) -> None:
    if not bool(getattr(params, "debug_stageB", False)):
        return
    f = _DEBUG_STAGEB_FRAME
    if f is None:
        print(f"[DEBUG stageB] :: {msg}", flush=True)
    else:
        print(f"[DEBUG stageB] frame={int(f)} :: {msg}", flush=True)


# -------------------------
# Core helpers
# -------------------------


def obs_vote(w0: float, w1: float, thr: float) -> Tuple[int, int]:
    return (1 if w0 > thr else 0, 1 if w1 > thr else 0)


def exclusive_mouse_id(v0: int, v1: int) -> Optional[int]:
    if v0 > 0 and v1 == 0:
        return 0
    if v1 > 0 and v0 == 0:
        return 1
    return None


def predict(prev1: Optional[np.ndarray], prev2: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if prev1 is None:
        return None
    if prev2 is None:
        return prev1
    return prev1 + (prev1 - prev2)


def penalty_for_exclusive(exclusive_label: Optional[int], assigned_mouse_id: int, penalty: float, bonus: float) -> float:
    if exclusive_label is None:
        return 0.0
    return -bonus if exclusive_label == assigned_mouse_id else penalty


# -------------------------
# Swap decision
# -------------------------


def decide_swap_state(prev_state: PrevState, anchors_now: Dict[str, List[TriangulatedItem]], params: StageBParams) -> str:
    # Determine whether we have any usable anchor continuity
    usable = False
    for p in params.anchor_parts:
        if len(anchors_now.get(p, [])) == 0:
            continue
        if prev_state.anchor_prev.get(p, {}).get(0) is not None or prev_state.anchor_prev.get(p, {}).get(1) is not None:
            usable = True
            break

    if not usable:
        _dbg_stageB(
            params,
            f"SWAP decide usable=False (no previous anchors). prev_swap={prev_state.swap_state} swap_age={int(prev_state.swap_age)} carry_max_gap={int(params.carry_swap_max_gap)}",
        )
        if prev_state.swap_age <= params.carry_swap_max_gap:
            return prev_state.swap_state
        return SwapState.UNKNOWN

    def cost_part(p: str, H: str) -> Tuple[float, int]:
        curr = anchors_now.get(p, [])
        if len(curr) == 0:
            _dbg_stageB(params, f"SWAP cost_part part={p} H={H} :: no current anchors")
            return (0.0, 0)

        if H == SwapState.NOSWAP:
            T0 = predict(prev_state.anchor_prev.get(p, {}).get(0), prev_state.anchor_prev2.get(p, {}).get(0))
            T1 = predict(prev_state.anchor_prev.get(p, {}).get(1), prev_state.anchor_prev2.get(p, {}).get(1))
        else:
            T0 = predict(prev_state.anchor_prev.get(p, {}).get(1), prev_state.anchor_prev2.get(p, {}).get(1))
            T1 = predict(prev_state.anchor_prev.get(p, {}).get(0), prev_state.anchor_prev2.get(p, {}).get(0))

        if T0 is None and T1 is None:
            _dbg_stageB(params, f"SWAP cost_part part={p} H={H} :: no targets (T0=None,T1=None)")
            return (0.0, 0)

        sig = params.sigma_motion_anchor

        # two current points and both targets exist
        if len(curr) >= 2 and T0 is not None and T1 is not None:
            item1, item2 = curr[0], curr[1]
            P1, P2 = item1.P, item2.P

            penA = penalty_for_exclusive(item1.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap) + \
                   penalty_for_exclusive(item2.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap)
            penB = penalty_for_exclusive(item1.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap) + \
                   penalty_for_exclusive(item2.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap)

            cA = (np.linalg.norm(P1 - T0) / sig) ** 2 + (np.linalg.norm(P2 - T1) / sig) ** 2 + penA
            cB = (np.linalg.norm(P1 - T1) / sig) ** 2 + (np.linalg.norm(P2 - T0) / sig) ** 2 + penB
            cmin = min(cA, cB)
            _dbg_stageB(
                params,
                (
                    f"SWAP cost_part part={p} H={H} :: curr=2 targets=2 sig={sig:.3f} "
                    f"P1=({P1[0]:.3f},{P1[1]:.3f},{P1[2]:.3f}) P2=({P2[0]:.3f},{P2[1]:.3f},{P2[2]:.3f}) "
                    f"T0=({T0[0]:.3f},{T0[1]:.3f},{T0[2]:.3f}) T1=({T1[0]:.3f},{T1[1]:.3f},{T1[2]:.3f}) "
                    f"penA={penA:.3f} penB={penB:.3f} cA={cA:.3f} cB={cB:.3f} -> min={cmin:.3f}"
                ),
            )
            return (cmin, 1)

        # one current point
        item = curr[0]
        P = item.P

        if T0 is not None and T1 is not None:
            pen0 = penalty_for_exclusive(item.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap)
            pen1 = penalty_for_exclusive(item.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap)
            d0 = (np.linalg.norm(P - T0) / sig) ** 2 + pen0
            d1 = (np.linalg.norm(P - T1) / sig) ** 2 + pen1
            dmin = min(d0, d1)
            _dbg_stageB(
                params,
                (
                    f"SWAP cost_part part={p} H={H} :: curr=1 targets=2 sig={sig:.3f} "
                    f"P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f}) "
                    f"T0=({T0[0]:.3f},{T0[1]:.3f},{T0[2]:.3f}) T1=({T1[0]:.3f},{T1[1]:.3f},{T1[2]:.3f}) "
                    f"d0={d0:.3f} d1={d1:.3f} -> min={dmin:.3f}"
                ),
            )
            return (dmin, 1)

        T = T0 if T1 is None else T1
        if T is None:
            _dbg_stageB(params, f"SWAP cost_part part={p} H={H} :: curr=1 no targets")
            return (0.0, 0)
        c = (np.linalg.norm(P - T) / sig) ** 2
        _dbg_stageB(
            params,
            (
                f"SWAP cost_part part={p} H={H} :: curr=1 targets=1 sig={sig:.3f} "
                f"P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f}) "
                f"T=({T[0]:.3f},{T[1]:.3f},{T[2]:.3f}) c={c:.3f}"
            ),
        )
        return (c, 1)

    C_no = 0.0
    C_sw = 0.0
    I_no = 0
    I_sw = 0

    for p in params.anchor_parts:
        c, i = cost_part(p, SwapState.NOSWAP)
        C_no += c
        I_no += i
        c, i = cost_part(p, SwapState.SWAP)
        C_sw += c
        I_sw += i

    _dbg_stageB(params, f"SWAP aggregate C_no={C_no:.3f} (I_no={I_no}) C_sw={C_sw:.3f} (I_sw={I_sw}) tau_swap={params.tau_swap:.3f}")

    if min(I_no, I_sw) == 0:
        if prev_state.swap_age <= params.carry_swap_max_gap:
            _dbg_stageB(params, "SWAP decide insufficient targets -> carry prev")
            return prev_state.swap_state
        _dbg_stageB(params, "SWAP decide insufficient targets -> UNKNOWN")
        return SwapState.UNKNOWN

    if abs(C_no - C_sw) < params.tau_swap:
        _dbg_stageB(params, f"SWAP decide ambiguous |C_no-C_sw|={abs(C_no-C_sw):.3f} < tau_swap={params.tau_swap:.3f}. prev_swap={prev_state.swap_state} age={int(prev_state.swap_age)}")
        if prev_state.swap_age <= params.carry_swap_max_gap:
            return prev_state.swap_state
        return SwapState.UNKNOWN

    out = SwapState.NOSWAP if C_no < C_sw else SwapState.SWAP
    _dbg_stageB(params, f"SWAP decide -> {out} (C_no={C_no:.3f}, C_sw={C_sw:.3f})")
    return out


# -------------------------
# Mouse models
# -------------------------


def build_mouse_models(
    prev_state: PrevState,
    anchors_now: Dict[str, List[TriangulatedItem]],
    swap_state: str,
    params: StageBParams,
) -> Tuple[Dict[int, Dict], Dict[str, Dict[int, Optional[np.ndarray]]], str]:
    """Assign current anchor points to mice using swap_state, then build centroid+axis models."""

    anchor_assigned: Dict[str, Dict[int, Optional[np.ndarray]]] = {p: {0: None, 1: None} for p in params.anchor_parts}

    for p in params.anchor_parts:
        curr = anchors_now.get(p, [])
        if curr:
            _dbg_stageB(params, f"MODEL anchors_now part={p} n={len(curr)} exclusives={[it.exclusive for it in curr[:2]]}")

    def targets_for(p: str, state: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if state == SwapState.NOSWAP:
            T0 = predict(prev_state.anchor_prev.get(p, {}).get(0), prev_state.anchor_prev2.get(p, {}).get(0))
            T1 = predict(prev_state.anchor_prev.get(p, {}).get(1), prev_state.anchor_prev2.get(p, {}).get(1))
        else:
            T0 = predict(prev_state.anchor_prev.get(p, {}).get(1), prev_state.anchor_prev2.get(p, {}).get(1))
            T1 = predict(prev_state.anchor_prev.get(p, {}).get(0), prev_state.anchor_prev2.get(p, {}).get(0))
        return T0, T1

    sig = params.sigma_motion_anchor

    for p in params.anchor_parts:
        curr = anchors_now.get(p, [])
        if len(curr) == 0:
            continue

        if swap_state != SwapState.UNKNOWN:
            T0, T1 = targets_for(p, swap_state)

            if len(curr) >= 2 and T0 is not None and T1 is not None:
                item1, item2 = curr[0], curr[1]
                P1, P2 = item1.P, item2.P

                penA = penalty_for_exclusive(item1.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap) + \
                       penalty_for_exclusive(item2.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap)
                penB = penalty_for_exclusive(item1.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap) + \
                       penalty_for_exclusive(item2.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap)

                cA = (np.linalg.norm(P1 - T0) / sig) ** 2 + (np.linalg.norm(P2 - T1) / sig) ** 2 + penA
                cB = (np.linalg.norm(P1 - T1) / sig) ** 2 + (np.linalg.norm(P2 - T0) / sig) ** 2 + penB

                if cA <= cB:
                    anchor_assigned[p][0] = P1
                    anchor_assigned[p][1] = P2
                else:
                    anchor_assigned[p][0] = P2
                    anchor_assigned[p][1] = P1

            else:
                # len==1 or missing targets
                item = curr[0]
                P = item.P
                if T0 is not None and T1 is not None:
                    d0 = (np.linalg.norm(P - T0) / sig) ** 2 + penalty_for_exclusive(item.exclusive, 0, params.penalty_excl_swap, params.bonus_excl_swap)
                    d1 = (np.linalg.norm(P - T1) / sig) ** 2 + penalty_for_exclusive(item.exclusive, 1, params.penalty_excl_swap, params.bonus_excl_swap)
                    if d0 <= d1:
                        anchor_assigned[p][0] = P
                    else:
                        anchor_assigned[p][1] = P
                else:
                    if T0 is not None:
                        anchor_assigned[p][0] = P
                    if T1 is not None:
                        anchor_assigned[p][1] = P
        else:
            # swap unknown: conservative - only use exclusives if they separate
            if len(curr) == 1 and curr[0].exclusive is not None:
                anchor_assigned[p][curr[0].exclusive] = curr[0].P
            elif len(curr) >= 2:
                e1, e2 = curr[0].exclusive, curr[1].exclusive
                if e1 in (0, 1) and e2 in (0, 1) and e1 != e2:
                    anchor_assigned[p][e1] = curr[0].P
                    anchor_assigned[p][e2] = curr[1].P

    for p in params.anchor_parts:
        if anchor_assigned.get(p, {}).get(0) is not None or anchor_assigned.get(p, {}).get(1) is not None:
            p0 = anchor_assigned[p][0]
            p1 = anchor_assigned[p][1]
            _dbg_stageB(
                params,
                f"MODEL anchor_assigned part={p} m0={(None if p0 is None else (float(p0[0]), float(p0[1]), float(p0[2])))} m1={(None if p1 is None else (float(p1[0]), float(p1[1]), float(p1[2])))}",
            )

    # Build mouse models
    mouse_model: Dict[int, Dict] = {0: {}, 1: {}}

    head_name, tail_name = params.anchor_parts[0], params.anchor_parts[1]

    for m in (0, 1):
        head = anchor_assigned.get(head_name, {}).get(m)
        tail = anchor_assigned.get(tail_name, {}).get(m)

        # fallback to previous anchors
        if head is None:
            head = prev_state.anchor_prev.get(head_name, {}).get(m)
        if tail is None:
            tail = prev_state.anchor_prev.get(tail_name, {}).get(m)

        # centroid
        if head is not None and tail is not None:
            centroid = 0.5 * (head + tail)
        elif head is not None:
            centroid = head
        elif tail is not None:
            centroid = tail
        else:
            centroid = prev_state.centroid_prev.get(m)

        axis_line = None
        valid_axis = False
        if head is not None and tail is not None:
            d = head - tail
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                axis_line = (tail, d / n)
                valid_axis = True

        if axis_line is None:
            axis_line = prev_state.axis_prev.get(m)
            valid_axis = axis_line is not None

        mouse_model[m] = {
            "centroid": centroid,
            "axis_line": axis_line,
            "head": head,
            "tail_root": tail,
            "valid_axis": valid_axis,
        }

        _dbg_stageB(
            params,
            f"MODEL mouse={m} swap_state={swap_state} head={(None if head is None else (float(head[0]), float(head[1]), float(head[2])))} "
            f"tail_root={(None if tail is None else (float(tail[0]), float(tail[1]), float(tail[2])))} "
            f"centroid={(None if centroid is None else (float(centroid[0]), float(centroid[1]), float(centroid[2])))} valid_axis={bool(valid_axis)}",
        )

    return mouse_model, anchor_assigned, swap_state


def _rebuild_mouse_models_from_anchor_assigned(
    prev_state: PrevState,
    anchor_assigned: Dict[str, Dict[int, Optional[np.ndarray]]],
    swap_state: str,
    params: StageBParams,
) -> Dict[int, Dict]:
    """Rebuild centroid+axis models from an already-decided anchor_assigned.

    This is used by B3.5 when we adjust the head<->tail_root pairing.
    """
    mouse_model: Dict[int, Dict] = {0: {}, 1: {}}

    head_name, tail_name = params.anchor_parts[0], params.anchor_parts[1]
    for m in (0, 1):
        head = anchor_assigned.get(head_name, {}).get(m)
        tail = anchor_assigned.get(tail_name, {}).get(m)

        # fallback to previous anchors
        if head is None:
            head = prev_state.anchor_prev.get(head_name, {}).get(m)
        if tail is None:
            tail = prev_state.anchor_prev.get(tail_name, {}).get(m)

        # centroid
        if head is not None and tail is not None:
            centroid = 0.5 * (head + tail)
        elif head is not None:
            centroid = head
        elif tail is not None:
            centroid = tail
        else:
            centroid = prev_state.centroid_prev.get(m)

        axis_line = None
        valid_axis = False
        if head is not None and tail is not None:
            d = head - tail
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                axis_line = (tail, d / n)
                valid_axis = True

        if axis_line is None:
            axis_line = prev_state.axis_prev.get(m)
            valid_axis = axis_line is not None

        mouse_model[m] = {
            "centroid": centroid,
            "axis_line": axis_line,
            "head": head,
            "tail_root": tail,
            "valid_axis": valid_axis,
        }

        _dbg_stageB(
            params,
            f"MODEL(B3.5) mouse={m} swap_state={swap_state} head={(None if head is None else (float(head[0]), float(head[1]), float(head[2])))} "
            f"tail_root={(None if tail is None else (float(tail[0]), float(tail[1]), float(tail[2])))} "
            f"centroid={(None if centroid is None else (float(centroid[0]), float(centroid[1]), float(centroid[2])))} valid_axis={bool(valid_axis)}",
        )

    return mouse_model


def _vote_contradiction(vote: Optional[Tuple[int, int]], assigned_mouse: int) -> int:
    if vote is None:
        return 0
    v0, v1 = vote
    if assigned_mouse == 0:
        return 1 if v1 > v0 else 0
    return 1 if v0 > v1 else 0


def _infer_perm_from_assigned(items2: List[TriangulatedItem], P_assigned_m0: Optional[np.ndarray]) -> str:
    """Infer whether anchor_assigned[m0] matches items2[0] or items2[1].

    Returns 'id' if items2[0] is closer to P_assigned_m0, else 'swap'.
    """
    if P_assigned_m0 is None or len(items2) < 2:
        return "id"
    d0 = float(np.linalg.norm(items2[0].P - P_assigned_m0))
    d1 = float(np.linalg.norm(items2[1].P - P_assigned_m0))
    return "id" if d0 <= d1 else "swap"


def apply_spine_check_B35(
    stageA_results: Dict[str, List[TriangulatedItem]],
    anchors_now: Dict[str, List[TriangulatedItem]],
    mouse_model: Dict[int, Dict],
    anchor_assigned: Dict[str, Dict[int, Optional[np.ndarray]]],
    prev_state: PrevState,
    swap_state: str,
    params: StageBParams,
) -> Tuple[Dict[int, Dict], Dict[str, Dict[int, Optional[np.ndarray]]]]:
    """B3.5: Head–tail_root edge sanity for two-mouse "half-stitch" failures.

    If a graph model is provided, we can validate the *within-mouse* head–tail_root
    distance and, if it is inconsistent, try alternative pairings of the two heads
    and the two tail_roots (swap-heads and/or swap-tails) to find a pairing that
    satisfies the edge-length ratio constraint.

    Selection priority:
      1) maximize number of mice whose head–tail_root edge passes the ratio test,
      2) minimize exclusive-segmentation penalties on the anchor assignments,
      3) minimize vote contradictions,
      4) be conservative w.r.t the original anchor_assigned (smallest movement).
    """

    if not params.graph_model:
        return mouse_model, anchor_assigned

    try:
        gm = _load_graph_model_cached(params.graph_model)
    except Exception as e:
        _dbg_stageB(params, f"B3.5 graph_model load failed: {e}")
        return mouse_model, anchor_assigned

    # Require this specific edge
    key_ht = tuple(sorted(("head", "tail_root")))
    if key_ht not in gm.edge_len:
        _dbg_stageB(params, "B3.5 skip: graph has no (head,tail_root) edge")
        return mouse_model, anchor_assigned

    heads = [it for it in anchors_now.get("head", []) if it.P is not None]
    tails = [it for it in anchors_now.get("tail_root", []) if it.P is not None]
    if len(heads) < 2 or len(tails) < 2:
        _dbg_stageB(params, f"B3.5 skip: need 2 heads + 2 tails (have heads={len(heads)}, tails={len(tails)})")
        return mouse_model, anchor_assigned

    H0, H1 = heads[0], heads[1]
    T0, T1 = tails[0], tails[1]

    perm_head_curr = _infer_perm_from_assigned([H0, H1], anchor_assigned.get("head", {}).get(0))
    perm_tail_curr = _infer_perm_from_assigned([T0, T1], anchor_assigned.get("tail_root", {}).get(0))

    def eval_combo(perm_head: str, perm_tail: str) -> Dict[str, float]:
        h_m0, h_m1 = (H0, H1) if perm_head == "id" else (H1, H0)
        t_m0, t_m1 = (T0, T1) if perm_tail == "id" else (T1, T0)

        ok0, L0, D0, r0 = _edge_ratio_check(gm, "head", "tail_root", h_m0.P, t_m0.P, params)
        ok1, L1, D1, r1 = _edge_ratio_check(gm, "head", "tail_root", h_m1.P, t_m1.P, params)
        ok_cnt = int(ok0) + int(ok1)

        # Anchor-exclusive penalties (segmentation consistency on anchors)
        seg_pen = 0.0
        seg_pen += penalty_for_exclusive(h_m0.exclusive, 0, params.penalty_excl, params.bonus_excl)
        seg_pen += penalty_for_exclusive(h_m1.exclusive, 1, params.penalty_excl, params.bonus_excl)
        seg_pen += penalty_for_exclusive(t_m0.exclusive, 0, params.penalty_excl, params.bonus_excl)
        seg_pen += penalty_for_exclusive(t_m1.exclusive, 1, params.penalty_excl, params.bonus_excl)

        # Vote contradictions on anchors
        contr = 0
        contr += _vote_contradiction(h_m0.vote, 0)
        contr += _vote_contradiction(h_m1.vote, 1)
        contr += _vote_contradiction(t_m0.vote, 0)
        contr += _vote_contradiction(t_m1.vote, 1)

        # Conservative: how far do we move from the current anchor_assigned?
        move = 0.0
        p_h0 = anchor_assigned.get("head", {}).get(0)
        p_h1 = anchor_assigned.get("head", {}).get(1)
        p_t0 = anchor_assigned.get("tail_root", {}).get(0)
        p_t1 = anchor_assigned.get("tail_root", {}).get(1)
        if p_h0 is not None:
            move += float(np.linalg.norm(h_m0.P - p_h0))
        if p_h1 is not None:
            move += float(np.linalg.norm(h_m1.P - p_h1))
        if p_t0 is not None:
            move += float(np.linalg.norm(t_m0.P - p_t0))
        if p_t1 is not None:
            move += float(np.linalg.norm(t_m1.P - p_t1))

        _dbg_stageB(
            params,
            (
                f"B3.5 combo head_perm={perm_head} tail_perm={perm_tail} :: "
                f"ok_cnt={ok_cnt} seg_pen={seg_pen:.3f} contr={contr} move={move:.3f} "
                f"m0(D={D0:.3f},ratio={r0:.3f},ok={ok0}) m1(D={D1:.3f},ratio={r1:.3f},ok={ok1})"
            ),
        )

        return {
            "ok_cnt": float(ok_cnt),
            "seg_pen": float(seg_pen),
            "contr": float(contr),
            "move": float(move),
            "perm_head": 0.0 if perm_head == "id" else 1.0,
            "perm_tail": 0.0 if perm_tail == "id" else 1.0,
        }

    combos = []
    for ph in ("id", "swap"):
        for pt in ("id", "swap"):
            combos.append((ph, pt, eval_combo(ph, pt)))

    # identify current
    curr = None
    for ph, pt, met in combos:
        if ph == perm_head_curr and pt == perm_tail_curr:
            curr = met
            break
    if curr is None:
        curr = combos[0][2]

    # pick best by priority
    best_ph, best_pt, best = sorted(
        combos,
        key=lambda x: (-x[2]["ok_cnt"], x[2]["seg_pen"], x[2]["contr"], x[2]["move"]),
    )[0]

    if (
        best_ph != perm_head_curr
        or best_pt != perm_tail_curr
    ) and (
        best["ok_cnt"] > curr["ok_cnt"]
        or (best["ok_cnt"] == curr["ok_cnt"] and best["seg_pen"] + 1e-9 < curr["seg_pen"])
    ):
        _dbg_stageB(
            params,
            (
                f"B3.5 apply head_perm={best_ph} tail_perm={best_pt} "
                f"(curr ok={int(curr['ok_cnt'])},seg_pen={curr['seg_pen']:.3f} -> best ok={int(best['ok_cnt'])},seg_pen={best['seg_pen']:.3f})"
            ),
        )

        h_m0, h_m1 = (H0, H1) if best_ph == "id" else (H1, H0)
        t_m0, t_m1 = (T0, T1) if best_pt == "id" else (T1, T0)
        anchor_assigned.setdefault("head", {0: None, 1: None})
        anchor_assigned.setdefault("tail_root", {0: None, 1: None})
        anchor_assigned["head"][0] = h_m0.P
        anchor_assigned["head"][1] = h_m1.P
        anchor_assigned["tail_root"][0] = t_m0.P
        anchor_assigned["tail_root"][1] = t_m1.P

        mouse_model = _rebuild_mouse_models_from_anchor_assigned(prev_state, anchor_assigned, swap_state, params)
    else:
        _dbg_stageB(params, f"B3.5 keep (head_perm={perm_head_curr}, tail_perm={perm_tail_curr})")

    return mouse_model, anchor_assigned


# -------------------------
# Assignment cost
# -------------------------


def assignment_cost(
    item: TriangulatedItem,
    mouse_id: int,
    bodypart: str,
    mouse_model: Dict[int, Dict],
    prev_state: PrevState,
    params: StageBParams,
) -> float:
    if item.P is None:
        return float("inf")

    P = item.P
    cost = 0.0

    term_excl = 0.0
    term_cont = 0.0
    term_cent = 0.0
    term_axis = 0.0

    # Exclusive evidence (strong during final assignment)
    if item.exclusive is not None:
        if item.exclusive == mouse_id:
            term_excl -= params.bonus_excl
        else:
            if params.hard_exclusive:
                return float("inf")
            term_excl += params.penalty_excl

    cost += term_excl

    # Continuity
    P1 = prev_state.prev.get(bodypart, {}).get(mouse_id)
    P0 = prev_state.prev2.get(bodypart, {}).get(mouse_id)
    if P1 is not None:
        Ppred = predict(P1, P0)
        sig = params.sigma_motion_anchor if bodypart in params.anchor_parts else params.sigma_motion_other
        term_cont = params.w_cont * (np.linalg.norm(P - Ppred) / sig) ** 2
        cost += term_cont

    # Centroid proximity
    centroid = mouse_model[mouse_id].get("centroid")
    if centroid is not None:
        term_cent = params.w_cent * (np.linalg.norm(P - centroid) / params.sigma_centroid) ** 2
        cost += term_cent

    # Axis proximity
    axis_line = mouse_model[mouse_id].get("axis_line")
    if axis_line is not None:
        line_p, line_d = axis_line
        d = distance_point_to_line(P, line_p, line_d)
        term_axis = params.w_axis * (d / params.sigma_axis) ** 2
        cost += term_axis

    _dbg_stageB(
        params,
        (
            f"COST bodypart={bodypart} item_kind={item.kind} assign_mouse={mouse_id} "
            f"P=({float(P[0]):.3f},{float(P[1]):.3f},{float(P[2]):.3f}) exclusive={item.exclusive} "
            f"terms={{excl:{term_excl:.3f},cont:{term_cont:.3f},cent:{term_cent:.3f},axis:{term_axis:.3f}}} -> cost={float(cost):.3f}"
        ),
    )

    return float(cost)


# -------------------------
# Mapping + singles
# -------------------------


def estimate_mapping_majority(pairs: List[Tuple[int, int]], params: Optional[StageBParams] = None, cam_name: Optional[str] = None) -> Dict[int, Optional[int]]:
    # pairs: list of (dlc_id, mouse_id)
    out: Dict[int, Optional[int]] = {0: None, 1: None}
    for dlc_id in (0, 1):
        m0 = sum(1 for d, m in pairs if d == dlc_id and m == 0)
        m1 = sum(1 for d, m in pairs if d == dlc_id and m == 1)
        if m0 > m1:
            out[dlc_id] = 0
        elif m1 > m0:
            out[dlc_id] = 1
        else:
            out[dlc_id] = None
        if params is not None and cam_name is not None:
            _dbg_stageB(params, f"B5 cam_map cam={cam_name} dlc_id={dlc_id} counts(m0={m0},m1={m1}) -> {out[dlc_id]}")
    return out


def continuity_fallback_single(
    obs: Observation,
    bodypart: str,
    rig: CameraRig,
    prev_state: PrevState,
    params: StageBParams,
) -> Optional[int]:
    cam = rig.cameras[obs.cam_idx]

    P0 = prev_state.prev.get(bodypart, {}).get(0)
    P1 = prev_state.prev.get(bodypart, {}).get(1)

    if P0 is None and P1 is None:
        _dbg_stageB(params, f"SINGLE continuity bodypart={bodypart} cam={obs.cam_name} dlc_id={obs.dlc_id} :: no prev points")
        return None

    d0 = float("inf")
    d1 = float("inf")

    if P0 is not None:
        u, v = cam.project(P0)
        if np.isfinite(u) and np.isfinite(v):
            d0 = float(np.hypot(u - obs.uv[0], v - obs.uv[1]))

    if P1 is not None:
        u, v = cam.project(P1)
        if np.isfinite(u) and np.isfinite(v):
            d1 = float(np.hypot(u - obs.uv[0], v - obs.uv[1]))

    if params.single_proj_margin_px > 0 and np.isfinite(d0) and np.isfinite(d1):
        if abs(d0 - d1) < params.single_proj_margin_px:
            _dbg_stageB(params, f"SINGLE continuity margin bodypart={bodypart} cam={obs.cam_name} -> None (|{d0:.3f}-{d1:.3f}|<{params.single_proj_margin_px:.3f})")
            return None

    out = 0 if d0 <= d1 else 1
    _dbg_stageB(params, f"SINGLE continuity bodypart={bodypart} cam={obs.cam_name} d0={d0:.3f} d1={d1:.3f} -> {out}")
    return out


# -------------------------
# B7: Graph plausibility check
# -------------------------


def _edge_ratio_check(gm: GraphModel, leaf: str, neighbor: str, Pleaf: np.ndarray, Pnbr: np.ndarray, params: StageBParams) -> Tuple[bool, float, float, float]:
    key = tuple(sorted((leaf, neighbor)))
    L = gm.edge_len.get(key, float("nan"))
    D = float(np.linalg.norm(Pleaf - Pnbr))
    if not np.isfinite(L) or L <= 1e-9 or not np.isfinite(D) or D <= 1e-9:
        return False, L, D, float("inf")
    ratio = float(L / D)
    ok = (params.graph_min_ratio <= ratio <= params.graph_max_ratio)
    return ok, L, D, ratio


def _collect_assigned_3d(stageA_results: Dict[str, List[TriangulatedItem]]) -> Dict[int, Dict[str, TriangulatedItem]]:
    """Collect a single representative 3D item per (mouse_id, bodypart).

    Stage A can output up to 2 triangulated hypotheses per bodypart (two mice),
    and those hypotheses can be of different kinds (triplet vs pair).

    For B7 we want a stable representative when fitting a rigid template and
    checking skeleton edges. We therefore choose:
      1) prefer triplet over pair (over any other kind), then
      2) if same kind: prefer smaller rms_px when available.
    """

    def _kind_rank(kind: str) -> int:
        k = str(kind)
        if k == "triplet":
            return 2
        if k == "pair":
            return 1
        return 0

    def _rms(it: TriangulatedItem) -> float:
        if it.stats is None:
            return float("inf")
        try:
            v = float(it.stats.rms_px)
        except Exception:
            return float("inf")
        return v if np.isfinite(v) else float("inf")

    assigned: Dict[int, Dict[str, TriangulatedItem]] = {0: {}, 1: {}}
    for b, items in stageA_results.items():
        for it in items:
            if it.P is None or it.mouse_id is None:
                continue
            m = int(it.mouse_id)
            if m not in (0, 1):
                continue
            prev = assigned[m].get(b)
            if prev is None:
                assigned[m][b] = it
                continue

            # Prefer triplet > pair > other
            r_new = _kind_rank(it.kind)
            r_old = _kind_rank(prev.kind)
            if r_new != r_old:
                if r_new > r_old:
                    assigned[m][b] = it
                continue

            # Same kind: prefer smaller rms_px if available
            if _rms(it) < _rms(prev):
                assigned[m][b] = it
    return assigned


def _fit_rigid_models(gm: GraphModel, assigned: Dict[int, Dict[str, TriangulatedItem]], params: StageBParams) -> Dict[int, Optional[Tuple[np.ndarray, np.ndarray, float]]]:
    fits: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, float]]] = {0: None, 1: None}
    for m in (0, 1):
        xs = []
        ys = []
        for name, it in assigned[m].items():
            if name not in gm.nodes:
                continue
            xs.append(gm.nodes[name])
            ys.append(it.P)
        if len(xs) < 3:
            _dbg_stageB(params, f"B7 rigid_fit mouse={m} :: insufficient points n={len(xs)}")
            fits[m] = None
            continue
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        fit = _kabsch_rigid(X, Y)
        if fit is None:
            _dbg_stageB(params, f"B7 rigid_fit mouse={m} :: failed")
        else:
            R, t, rms = fit
            _dbg_stageB(params, f"B7 rigid_fit mouse={m} :: n={len(xs)} rms={rms:.3f}")
        fits[m] = fit
    return fits


def _rigid_leaf_error(gm: GraphModel, fit: Tuple[np.ndarray, np.ndarray, float], leaf: str, Pleaf: np.ndarray) -> float:
    R, t, _rms = fit
    x = gm.nodes.get(leaf)
    if x is None:
        return float("inf")
    # world_pred = R @ x + t
    world_pred = (R @ x) + t
    return float(np.linalg.norm(Pleaf - world_pred))


def _leaf_tol_mm(gm: GraphModel, leaf: str, params: StageBParams) -> float:
    # if we can find an incident edge, scale tol by that edge length
    nbrs = gm.neighbors.get(leaf, [])
    L = None
    if nbrs:
        key = tuple(sorted((leaf, nbrs[0])))
        L = gm.edge_len.get(key)
    if L is None or not np.isfinite(L) or L <= 0:
        return float(params.graph_rigid_tol_mm_max)
    return float(min(params.graph_rigid_tol_mm_max, params.graph_rigid_tol_ratio * float(L)))


def apply_graph_check_B7(
    stageA_results: Dict[str, List[TriangulatedItem]],
    params: StageBParams,
) -> None:
    """Optional post-check after B6.

    - Only runs if params.graph_model is set.
    - Operates on current-frame 3D items (P != None).
    """
    if not params.graph_model:
        return

    try:
        gm = _load_graph_model_cached(params.graph_model)
    except Exception as e:
        _dbg_stageB(params, f"B7 graph_model load failed: {e}")
        return

    leaf_nodes = gm.leaf_nodes() if params.graph_leaf_only else list(gm.nodes.keys())
    _dbg_stageB(params, f"B7 graph_model='{params.graph_model}' leaf_only={bool(params.graph_leaf_only)} leaves={leaf_nodes}")

    assigned = _collect_assigned_3d(stageA_results)

    # Determine initial leaf status and which ones violate edge constraints.
    freed: List[Tuple[str, TriangulatedItem, int, str]] = []
    # tuple: (bodypart, item, current_mouse, reason)

    def neighbor_for(name: str) -> Optional[str]:
        nbs = gm.neighbors.get(name, [])
        if len(nbs) == 1:
            return nbs[0]
        return None

    # Pass 1: free leaves that violate edge constraints (when neighbor is present)
    for m in (0, 1):
        for leaf in leaf_nodes:
            it = assigned[m].get(leaf)
            if it is None or it.P is None or it.mouse_id is None:
                continue
            nbr = neighbor_for(leaf)
            if nbr is None:
                continue
            it_n = assigned[m].get(nbr)
            if it_n is None or it_n.P is None:
                # disconnected: handle later via rigid fallback / other-mouse check
                freed.append((leaf, it, m, "no_neighbor"))
                continue
            ok, L, D, ratio = _edge_ratio_check(gm, leaf, nbr, it.P, it_n.P, params)
            _dbg_stageB(params, f"B7 edge_check m={m} leaf={leaf} nbr={nbr} L={L:.3f} D={D:.3f} ratio={ratio:.3f} ok={ok}")
            if not ok:
                freed.append((leaf, it, m, "violation"))

    # Free them (temporarily)
    for leaf, it, m, reason in freed:
        _dbg_stageB(params, f"B7 free leaf={leaf} from m={m} reason={reason}")
        it.mouse_id = None

    # Rebuild assigned after freeing
    assigned = _collect_assigned_3d(stageA_results)

    # Pass 2: try assigning freed leaves to the other mouse using edge check
    still_free: List[Tuple[str, TriangulatedItem, int, str]] = []
    for leaf, it, m_old, reason in freed:
        if it.P is None:
            continue
        other = 1 - m_old

        # do not create duplicates
        if leaf in assigned.get(other, {}):
            still_free.append((leaf, it, m_old, reason))
            _dbg_stageB(params, f"B7 try_other leaf={leaf} m_old={m_old} -> other has leaf already; keep free")
            continue

        nbr = neighbor_for(leaf)
        if nbr is not None:
            it_n = assigned[other].get(nbr)
            if it_n is not None and it_n.P is not None:
                ok, L, D, ratio = _edge_ratio_check(gm, leaf, nbr, it.P, it_n.P, params)
                _dbg_stageB(params, f"B7 try_other edge m_new={other} leaf={leaf} nbr={nbr} L={L:.3f} D={D:.3f} ratio={ratio:.3f} ok={ok}")
                if ok:
                    it.mouse_id = other
                    assigned[other][leaf] = it
                    _dbg_stageB(params, f"B7 reassigned leaf={leaf} -> m={other} by edge_check")
                    continue

        still_free.append((leaf, it, m_old, reason))

    # Pass 3: rigid fallback for remaining free leaves
    # Fit rigid per mouse from currently assigned 3D points.
    fits = _fit_rigid_models(gm, assigned, params)

    for leaf, it, m_old, reason in still_free:
        if it.P is None:
            continue

        # compute errors
        e0 = _rigid_leaf_error(gm, fits[0], leaf, it.P) if fits[0] is not None else float("inf")
        e1 = _rigid_leaf_error(gm, fits[1], leaf, it.P) if fits[1] is not None else float("inf")
        tol = _leaf_tol_mm(gm, leaf, params)

        _dbg_stageB(params, f"B7 rigid_try leaf={leaf} err0={e0:.3f} err1={e1:.3f} tol={tol:.3f} reason={reason}")

        # choose best
        best_m = 0 if e0 <= e1 else 1
        best_e = min(e0, e1)

        if np.isfinite(best_e) and best_e <= tol:
            # avoid duplicates
            if leaf in assigned.get(best_m, {}):
                _dbg_stageB(params, f"B7 rigid_assign leaf={leaf} best_m={best_m} but duplicate exists -> drop")
            else:
                it.mouse_id = best_m
                assigned[best_m][leaf] = it
                _dbg_stageB(params, f"B7 rigid_assign leaf={leaf} -> m={best_m} err={best_e:.3f}")
                continue

        # undecidable: decide keep vs drop
        if reason == "violation":
            # it failed edge check and couldn't be rescued
            it.mouse_id = None
            _dbg_stageB(params, f"B7 drop leaf={leaf} (failed edge and rigid)")
        else:
            # disconnected: be conservative unless configured otherwise
            if params.graph_drop_if_undecidable:
                it.mouse_id = None
                _dbg_stageB(params, f"B7 drop leaf={leaf} (undecidable+graph_drop_if_undecidable)")
            else:
                it.mouse_id = m_old
                _dbg_stageB(params, f"B7 keep leaf={leaf} with original m={m_old} (undecidable)")


def apply_head_tailroot_leafdrop_B72(
    stageA_results: Dict[str, List[TriangulatedItem]],
    anchor_assigned: Dict[str, Dict[int, Optional[np.ndarray]]],
    params: StageBParams,
) -> List[int]:
    """B7.2: Drop tail_root if it is *effectively a leaf* and violates the head↔tail_root edge.

    Rationale
    - Stage A can occasionally accept a low-RMS pairwise triangulation built from mismatched
      observations across cameras (e.g., a tail_root from mouse A in one view + mouse B in another).
      If that tail_root ends up isolated (no hind paws / tail tip assigned to the same mouse), the
      standard B7 leaf checks cannot flag it.
    - Here we treat tail_root as a "factual leaf" when none of its non-head neighbors are present
      for the same mouse, and then apply the head-tail_root edge length ratio test.
    - If the ratio test fails, we *drop only that tail_root* and do **not** try to reassign it to the
      other mouse (B3.5 already explored anchor permutations).

    Returns: list of mice (0/1) for which tail_root was dropped.
    """

    if not params.graph_model:
        return []

    try:
        gm = _load_graph_model_cached(params.graph_model)
    except Exception as e:
        _dbg_stageB(params, f"B7.2 graph_model load failed: {e}")
        return []

    if "head" not in gm.nodes or "tail_root" not in gm.nodes:
        return []

    assigned = _collect_assigned_3d(stageA_results)
    dropped: List[int] = []

    for m in (0, 1):
        it_h = assigned[m].get("head")
        it_t = assigned[m].get("tail_root")
        if it_h is None or it_t is None or it_h.P is None or it_t.P is None:
            continue

        # tail_root is "effectively a leaf" if none of its other neighbors are present.
        nbrs = set(gm.neighbors.get("tail_root", []))
        nbrs.discard("head")
        # Only consider bodyparts we actually track.
        nbrs = {n for n in nbrs if n in stageA_results}
        has_other = any((assigned[m].get(n) is not None and assigned[m][n].P is not None) for n in nbrs)
        effective_leaf = not has_other

        ok, L, D, ratio = _edge_ratio_check(gm, "head", "tail_root", it_h.P, it_t.P, params)
        _dbg_stageB(
            params,
            (
                f"B7.2 head-tail_root m={m} effective_leaf={bool(effective_leaf)} "
                f"other_nbrs={sorted(list(nbrs))} ok={bool(ok)} "
                f"L={L:.3f} D={D:.3f} ratio={ratio:.3f}"
            ),
        )

        if not effective_leaf or ok:
            continue

        # Drop only this tail_root (do not try other).
        for it in stageA_results.get("tail_root", []):
            if it.P is None:
                continue
            if it.mouse_id == m:
                it.mouse_id = None

        anchor_assigned.setdefault("tail_root", {0: None, 1: None})
        anchor_assigned["tail_root"][m] = None
        dropped.append(m)
        _dbg_stageB(params, f"B7.2 drop tail_root m={m} reason=head-tail_root_violation")

    return dropped


# -------------------------
# Stage B main
# -------------------------


def stage_b_assign(
    frame: int,
    rig: CameraRig,
    stageA_results: Dict[str, List[TriangulatedItem]],
    seg_source: Optional[SegmentationSource],
    prev_state: PrevState,
    params: StageBParams,
) -> Tuple[Dict[str, List[TriangulatedItem]], PrevState]:

    global _DEBUG_STAGEB_FRAME
    _DEBUG_STAGEB_FRAME = int(frame)

    _dbg_stageB(
        params,
        (
            f"START seg_source={'present' if seg_source is not None else 'none'} "
            f"vote_thr={params.vote_thr:.4f} seg_radius={int(params.seg_radius)} seg_sigma={params.seg_sigma:.3f} "
            f"prev_swap={prev_state.swap_state} swap_age={int(prev_state.swap_age)}"
        ),
    )

    kernel = GaussianKernel.make(radius=params.seg_radius, sigma=params.seg_sigma)

    # B1: compute seg weights + votes per obs and per item
    if seg_source is None:
        # no segmentation available -> all votes empty
        for b, items in stageA_results.items():
            for item in items:
                item.vote = (0, 0)
                item.exclusive = None
        anchors_now = {p: [it for it in stageA_results.get(p, []) if it.P is not None] for p in params.anchor_parts}
    else:
        # cache masks per cam for this frame
        masks: Dict[str, np.ndarray] = {}
        for cam in rig.cameras:
            mask = seg_source.get_mask(cam.name, frame)
            masks[cam.name] = mask
            if bool(params.debug_stageB):
                # print label histogram for {0,1,2} only
                vals, cnts = np.unique(mask.astype(np.int32), return_counts=True)
                counts = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}
                c0 = counts.get(0, 0)
                c1 = counts.get(1, 0)
                c2 = counts.get(2, 0)
                _dbg_stageB(params, f"B1 mask cam={cam.name} shape={mask.shape} unique=0:{c0},1:{c1},2:{c2}")

        for b, items in stageA_results.items():
            for item in items:
                v0 = 0
                v1 = 0
                for obs in item.obs_list:
                    if obs.seg_w is None:
                        mask = masks[obs.cam_name]
                        obs.seg_w = seg_weights_from_mask(mask, obs.uv, kernel)
                    wbg, w0, w1w = obs.seg_w
                    dv0, dv1 = obs_vote(w0, w1w, params.vote_thr)
                    obs.vote = (dv0, dv1)
                    v0 += dv0
                    v1 += dv1
                    _dbg_stageB(
                        params,
                        (
                            f"B1 obs bodypart={b} item_kind={item.kind} cam={obs.cam_name} dlc_id={obs.dlc_id} "
                            f"uv=({float(obs.uv[0]):.3f},{float(obs.uv[1]):.3f}) "
                            f"seg_w(bg={wbg:.3f},m0={w0:.3f},m1={w1w:.3f}) vote=({dv0},{dv1}) thr={params.vote_thr:.4f}"
                        ),
                    )
                item.vote = (v0, v1)
                item.exclusive = exclusive_mouse_id(v0, v1)
                _dbg_stageB(
                    params,
                    (
                        f"B1 item bodypart={b} kind={item.kind} num_obs={len(item.obs_list)} "
                        f"vote_sum=({v0},{v1}) exclusive={item.exclusive}"
                    ),
                )

        anchors_now = {p: [it for it in stageA_results.get(p, []) if it.P is not None] for p in params.anchor_parts}

    # B2: swap decision
    swap_state = decide_swap_state(prev_state, anchors_now, params)
    _dbg_stageB(params, f"B2 swap_state={swap_state}")

    # B3: build mouse models
    mouse_model, anchor_assigned, swap_state = build_mouse_models(prev_state, anchors_now, swap_state, params)
    _dbg_stageB(params, f"B3 built mouse models (swap_state={swap_state})")

    # B3.5 (optional): graph-based head<->tail_root pairing sanity (prevents half-stitch chimeras)
    if params.graph_model:
        mouse_model, anchor_assigned = apply_spine_check_B35(
            stageA_results,
            anchors_now,
            mouse_model,
            anchor_assigned,
            prev_state,
            swap_state,
            params,
        )

    
    # B3.6: lock anchor assignments after B3.5
    #
    # B3/B3.5 determines a consistent head<->tail_root pairing and builds the per-mouse centroid/axis
    # model from those anchors. Without locking, B4 can still swap anchor parts based on continuity
    # costs, reintroducing the "chimera" (front of one mouse + back of the other). We therefore lock
    # the anchor items' mouse_id to the pairing implied by anchor_assigned.
    locked_anchor_parts = set()

    def _lock_anchor_part(p: str) -> None:
        pts = [it for it in stageA_results.get(p, []) if it.P is not None]
        if len(pts) == 0:
            return
        P0 = anchor_assigned.get(p, {}).get(0)
        P1 = anchor_assigned.get(p, {}).get(1)
        if P0 is None and P1 is None:
            return

        # If we have two anchor targets, choose the best 1-1 pairing for the first two points.
        if len(pts) >= 2 and P0 is not None and P1 is not None:
            A, B = pts[0], pts[1]
            dA0 = float(np.linalg.norm(A.P - P0))
            dA1 = float(np.linalg.norm(A.P - P1))
            dB0 = float(np.linalg.norm(B.P - P0))
            dB1 = float(np.linalg.norm(B.P - P1))
            if dA0 + dB1 <= dA1 + dB0:
                A.mouse_id, B.mouse_id = 0, 1
            else:
                A.mouse_id, B.mouse_id = 1, 0
            locked_anchor_parts.add(p)
            _dbg_stageB(
                params,
                f"B3.6 lock_anchor part={p} :: A->m{A.mouse_id} B->m{B.mouse_id} "
                f"dA0={dA0:.3f} dA1={dA1:.3f} dB0={dB0:.3f} dB1={dB1:.3f}",
            )

            # Any additional 3D points (rare) are assigned to the nearest anchor target.
            for extra in pts[2:]:
                m = 0 if float(np.linalg.norm(extra.P - P0)) <= float(np.linalg.norm(extra.P - P1)) else 1
                extra.mouse_id = m
                _dbg_stageB(params, f"B3.6 lock_anchor part={p} :: extra kind={extra.kind} -> m{m}")
            return

        # Only one anchor target or only one point: assign by nearest available target.
        for it in pts:
            if P0 is None:
                it.mouse_id = 1
            elif P1 is None:
                it.mouse_id = 0
            else:
                it.mouse_id = 0 if float(np.linalg.norm(it.P - P0)) <= float(np.linalg.norm(it.P - P1)) else 1
            locked_anchor_parts.add(p)
            _dbg_stageB(params, f"B3.6 lock_anchor part={p} :: kind={it.kind} -> m{it.mouse_id}")

    for _p in params.anchor_parts:
        _lock_anchor_part(_p)

# B4: assign triangulated 3D items
    for b, items in stageA_results.items():
        if b in locked_anchor_parts:
            _dbg_stageB(params, f"B4 skip bodypart={b} (locked anchor)")
            continue
        points3D = [it for it in items if it.P is not None]
        if len(points3D) == 2:
            A, B = points3D[0], points3D[1]

            cA0 = assignment_cost(A, 0, b, mouse_model, prev_state, params)
            cA1 = assignment_cost(A, 1, b, mouse_model, prev_state, params)
            cB0 = assignment_cost(B, 0, b, mouse_model, prev_state, params)
            cB1 = assignment_cost(B, 1, b, mouse_model, prev_state, params)

            c01 = cA0 + cB1
            c10 = cA1 + cB0

            _dbg_stageB(
                params,
                (
                    f"B4 assign2 bodypart={b} :: A(excl={A.exclusive},vote={A.vote}) B(excl={B.exclusive},vote={B.vote}) "
                    f"cA0={cA0:.3f} cA1={cA1:.3f} cB0={cB0:.3f} cB1={cB1:.3f} "
                    f"c01(A->0,B->1)={c01:.3f} c10(A->1,B->0)={c10:.3f}"
                ),
            )

            if c01 <= c10:
                A.mouse_id = 0
                B.mouse_id = 1
            else:
                A.mouse_id = 1
                B.mouse_id = 0

            _dbg_stageB(params, f"B4 result bodypart={b} :: A.mouse_id={A.mouse_id} B.mouse_id={B.mouse_id}")

        elif len(points3D) == 1:
            A = points3D[0]
            if A.exclusive is not None:
                A.mouse_id = A.exclusive
                _dbg_stageB(params, f"B4 assign1 bodypart={b} :: exclusive -> {A.mouse_id}")
            else:
                c0 = assignment_cost(A, 0, b, mouse_model, prev_state, params)
                c1 = assignment_cost(A, 1, b, mouse_model, prev_state, params)
                A.mouse_id = 0 if c0 <= c1 else 1
                _dbg_stageB(params, f"B4 assign1 bodypart={b} :: c0={c0:.3f} c1={c1:.3f} -> {A.mouse_id}")

    # B5: per-camera dlc_id -> new mouse mapping from resolved items
    cam_maps: Dict[str, Dict[int, Optional[int]]] = {}
    for cam in rig.cameras:
        pairs: List[Tuple[int, int]] = []
        for b, items in stageA_results.items():
            for item in items:
                if item.mouse_id is None:
                    continue
                for obs in item.obs_list:
                    if obs.cam_name == cam.name:
                        pairs.append((obs.dlc_id, int(item.mouse_id)))
        cam_maps[cam.name] = estimate_mapping_majority(pairs, params=params, cam_name=cam.name)

    # B6: assign singles
    for b, items in stageA_results.items():
        for item in items:
            if item.kind != "single":
                continue
            obs = item.obs_list[0]

            # 1) exclusive segmentation decisive
            if item.exclusive is not None:
                item.mouse_id = item.exclusive
                _dbg_stageB(params, f"B6 single bodypart={b} cam={obs.cam_name} dlc_id={obs.dlc_id} uv=({obs.uv[0]:.3f},{obs.uv[1]:.3f}) exclusive -> {item.mouse_id}")
                continue

            v0, v1 = item.vote if item.vote is not None else (0, 0)

            # 2) only if no segmentation signal at all
            if v0 == 0 and v1 == 0:
                mapped = cam_maps.get(obs.cam_name, {}).get(obs.dlc_id)
                if mapped is not None:
                    item.mouse_id = mapped
                    _dbg_stageB(params, f"B6 single bodypart={b} cam={obs.cam_name} dlc_id={obs.dlc_id} uv=({obs.uv[0]:.3f},{obs.uv[1]:.3f}) vote=(0,0) -> cam_map {mapped}")
                    continue

            # 3) continuity via previous 3D projection
            mid = continuity_fallback_single(obs, b, rig, prev_state, params)
            item.mouse_id = mid
            _dbg_stageB(params, f"B6 single bodypart={b} cam={obs.cam_name} dlc_id={obs.dlc_id} uv=({obs.uv[0]:.3f},{obs.uv[1]:.3f}) -> continuity {mid}")

    # B7 (optional): graph plausibility post-check
    if params.graph_model:
        apply_graph_check_B7(stageA_results, params)

        # B7.2: If tail_root is effectively isolated (missing its other neighbors), apply
        # head-tail_root edge check and drop only the offending tail_root.
        dropped_tail = apply_head_tailroot_leafdrop_B72(stageA_results, anchor_assigned, params)
        if dropped_tail:
            mouse_model = _rebuild_mouse_models_from_anchor_assigned(prev_state, anchor_assigned, swap_state, params)

    # Update prev_state
    new_state = PrevState(
        prev={k: {0: (v.get(0).copy() if v.get(0) is not None else None), 1: (v.get(1).copy() if v.get(1) is not None else None)} for k, v in prev_state.prev.items()},
        prev2={k: {0: (v.get(0).copy() if v.get(0) is not None else None), 1: (v.get(1).copy() if v.get(1) is not None else None)} for k, v in prev_state.prev2.items()},
        anchor_prev={k: {0: (v.get(0).copy() if v.get(0) is not None else None), 1: (v.get(1).copy() if v.get(1) is not None else None)} for k, v in prev_state.anchor_prev.items()},
        anchor_prev2={k: {0: (v.get(0).copy() if v.get(0) is not None else None), 1: (v.get(1).copy() if v.get(1) is not None else None)} for k, v in prev_state.anchor_prev2.items()},
        centroid_prev={0: mouse_model[0]["centroid"], 1: mouse_model[1]["centroid"]},
        axis_prev={0: mouse_model[0]["axis_line"], 1: mouse_model[1]["axis_line"]},
        swap_state=swap_state,
        swap_age=(0 if swap_state != SwapState.UNKNOWN else prev_state.swap_age + 1),
    )

    # shift prev2 <- prev, prev <- current assignments
    # initialize dicts
    for b in stageA_results.keys():
        if b not in new_state.prev:
            new_state.prev[b] = {0: None, 1: None}
        if b not in new_state.prev2:
            new_state.prev2[b] = {0: None, 1: None}

        # shift
        new_state.prev2[b][0] = prev_state.prev.get(b, {}).get(0)
        new_state.prev2[b][1] = prev_state.prev.get(b, {}).get(1)

        # set current
        for item in stageA_results[b]:
            if item.P is None or item.mouse_id is None:
                continue
            new_state.prev[b][int(item.mouse_id)] = item.P

    # anchors
    for p in params.anchor_parts:
        if p not in new_state.anchor_prev:
            new_state.anchor_prev[p] = {0: None, 1: None}
        if p not in new_state.anchor_prev2:
            new_state.anchor_prev2[p] = {0: None, 1: None}

        new_state.anchor_prev2[p][0] = prev_state.anchor_prev.get(p, {}).get(0)
        new_state.anchor_prev2[p][1] = prev_state.anchor_prev.get(p, {}).get(1)

        new_state.anchor_prev[p][0] = _coalesce(anchor_assigned.get(p, {}).get(0), new_state.prev.get(p, {}).get(0))
        new_state.anchor_prev[p][1] = _coalesce(anchor_assigned.get(p, {}).get(1), new_state.prev.get(p, {}).get(1))

    _dbg_stageB(params, f"END swap_state={swap_state} swap_age={int(new_state.swap_age)}")
    _DEBUG_STAGEB_FRAME = None

    return stageA_results, new_state
