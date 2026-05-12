import json
import itertools
import numpy as np
import pandas as pd

import ceres_point_ba  # your compiled module

def load_rig(cameras_json):
    with open(cameras_json, "r") as f:
        j = json.load(f)
    cams = j["cameras"]
    cams_sorted = sorted(cams, key=lambda c: c["name"])
    C = len(cams_sorted)
    fx = np.array([c["fx"] for c in cams_sorted], float)
    fy = np.array([c["fy"] for c in cams_sorted], float)
    cx = np.array([c["cx"] for c in cams_sorted], float)
    cy = np.array([c["cy"] for c in cams_sorted], float)
    R = np.array([c["R"] for c in cams_sorted], float)     # world->cam
    t = np.array([c["t"] for c in cams_sorted], float)     # world->cam
    names = [c["name"] for c in cams_sorted]
    return {"fx":fx,"fy":fy,"cx":cx,"cy":cy,"R":R,"t":t,"names":names}

def cam_center_world(R, t):
    # Pc = R Pw + t  =>  C = -R^T t
    return -R.T @ t

def ray_world_from_uv(rig, cam_idx, uv):
    fx,fy,cx,cy = rig["fx"][cam_idx], rig["fy"][cam_idx], rig["cx"][cam_idx], rig["cy"][cam_idx]
    R = rig["R"][cam_idx]
    t = rig["t"][cam_idx]
    x = (uv[0]-cx)/fx
    y = (uv[1]-cy)/fy
    d_c = np.array([x,y,1.0], float)
    d_c /= np.linalg.norm(d_c)
    d_w = R.T @ d_c
    d_w /= np.linalg.norm(d_w)
    C = cam_center_world(R, t)
    return C, d_w

def closest_midpoint_two_rays(C1,d1,C2,d2):
    # returns midpoint between closest points on two skew lines
    w0 = C1 - C2
    a = np.dot(d1,d1)
    b = np.dot(d1,d2)
    c = np.dot(d2,d2)
    d = np.dot(d1,w0)
    e = np.dot(d2,w0)
    denom = a*c - b*b
    if abs(denom) < 1e-12:
        # nearly parallel: pick arbitrary
        s = 0.0
        t = e/c if c>1e-12 else 0.0
    else:
        s = (b*e - c*d)/denom
        t = (a*e - b*d)/denom
    P1 = C1 + s*d1
    P2 = C2 + t*d2
    return 0.5*(P1+P2), np.linalg.norm(P1-P2)

def angle_deg(d1,d2):
    c = np.clip(np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def solve_point(P_init, cam_idx, uv, rig, sigma_pix=2.0):
    cams = {"fx": rig["fx"], "fy": rig["fy"], "cx": rig["cx"], "cy": rig["cy"], "R": rig["R"], "t": rig["t"]}
    P_ref, stats = ceres_point_ba.solve_point(
        np.asarray(P_init, float),
        list(map(int, cam_idx)),
        np.asarray(uv, float),
        cams,
        sigmas=None,
        sigma_pix=float(sigma_pix),
        max_num_iterations=50,
        loss="huber",
        loss_scale=1.0,
        verbose=False,
    )
    return np.asarray(P_ref, float), stats

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--dlc", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--bodypart", required=True)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--max-rms", type=float, default=6.0)
    args = ap.parse_args()

    rig = load_rig(args.cameras)
    df = pd.read_csv(args.dlc)
    df = df[(df["frame"]==args.frame) & (df["bodypart"]==args.bodypart)]
    df = df[df["x"].notna() & df["y"].notna()]

    # build candidates per camera: list of (dlc_id, uv)
    cam_to_cands = {name: [] for name in rig["names"]}
    for _,r in df.iterrows():
        cam = r["camera"]
        if cam in cam_to_cands:
            cam_to_cands[cam].append((int(r["individual"].replace("mouse","")) if isinstance(r["individual"], str) else int(r["individual"]),
                                      np.array([r["x"], r["y"]], float)))

    # map names->idx
    name_to_idx = {n:i for i,n in enumerate(rig["names"])}

    cams_present = [c for c,v in cam_to_cands.items() if len(v)>0]
    print("cams_present:", cams_present, "counts:", {c:len(cam_to_cands[c]) for c in cams_present})
    if len(cams_present) < 2:
        print("Not enough cameras.")
        return

    # enumerate triplets if 3 cams present, else pairs only
    best = []
    cam_trip = [c for c in rig["names"] if len(cam_to_cands[c])>0]
    # limit to first 3 cameras if more (should be exactly 3)
    cam_trip = cam_trip[:3]

    def test_combo(cam_names, choice):
        cam_idx = []
        uv = []
        rays = []
        for cam_name,(dlc_id,uv_i) in zip(cam_names, choice):
            ci = name_to_idx[cam_name]
            cam_idx.append(ci)
            uv.append(uv_i)
            C,d = ray_world_from_uv(rig, ci, uv_i)
            rays.append((C,d))
        # init from best pair among those rays
        best_init = None
        best_gap = 1e18
        for (C1,d1),(C2,d2) in itertools.combinations(rays,2):
            P0,gap = closest_midpoint_two_rays(C1,d1,C2,d2)
            if gap < best_gap:
                best_gap = gap
                best_init = P0
        P_ref, stats = solve_point(best_init, cam_idx, uv, rig, sigma_pix=args.sigma)
        return {
            "cams": cam_names,
            "dlc_ids": [x[0] for x in choice],
            "rms": float(stats["rms_px"]),
            "max": float(stats["max_err_px"]),
            "gap_init_mm": float(best_gap),
            "P": P_ref,
        }

    # triplets
    if len(cam_trip) == 3:
        combos = list(itertools.product(*[cam_to_cands[c] for c in cam_trip]))
        print("Triplet combos:", len(combos))
        for choice in combos:
            res = test_combo(cam_trip, choice)
            best.append(res)

    # pairs
    for camA, camB in itertools.combinations([c for c in rig["names"] if len(cam_to_cands[c])>0], 2):
        combos = list(itertools.product(cam_to_cands[camA], cam_to_cands[camB]))
        print(f"Pair combos {camA}-{camB}:", len(combos))
        for choice in combos:
            res = test_combo([camA, camB], choice)
            best.append(res)

    best.sort(key=lambda r: r["rms"])
    print("\nTop 10 by RMS:")
    for r in best[:10]:
        ok = "OK" if r["rms"] <= args.max_rms else "bad"
        print(ok, "cams", r["cams"], "dlc_ids", r["dlc_ids"], "rms", r["rms"], "max", r["max"], "gap_init_mm", r["gap_init_mm"])

if __name__ == "__main__":
    main()
