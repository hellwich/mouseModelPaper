import json
import numpy as np
import pandas as pd
import imageio.v3 as iio

from mice3d.segmentation import GaussianKernel, seg_weights_from_mask

# ---- Paths (match your repo layout) ----
SEG_ROOT = "../../mouse_seg/preds/out2m_2"  # contains top/front/side subdirs
DLC_CSV = "../../mouse_sim/out2m_2_render/dlc_long.csv"
CAMERAS_JSON = "../../mouse_sim/out2m_2/cameras.json"

FRAME = 0
BODYPART = "head"  # change if you want to test a different bodypart

# Map camera name (in DLC/cameras.json) -> segmentation subdir name
CAM_TO_VIEWDIR = {
    "cam1_top": "top",
    "cam2_front": "front",
    "cam3_side": "side",
}


def load_camera_sizes(cameras_json_path: str) -> dict:
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sizes = {}
    for c in data.get("cameras", []):
        name = str(c.get("name"))
        w = int(c.get("width"))
        h = int(c.get("height"))
        sizes[name] = (w, h)
    return sizes


def pick_uvs(df: pd.DataFrame, frame: int, bodypart: str) -> dict:
    """Return dict cam_name -> list of (individual, uv) for the given frame+bodypart."""
    out = {}
    sub = df[(df["frame"] == frame) & (df["bodypart"] == bodypart)].copy()
    for cam in sorted(sub["camera"].unique()):
        out[cam] = []
        for ind in ["mouse0", "mouse1"]:
            r = sub[(sub["camera"] == cam) & (sub["individual"] == ind)]
            if len(r) == 0:
                continue
            row = r.iloc[0]
            out[cam].append((ind, np.array([float(row.x), float(row.y)], dtype=float)))
    return out


def test_one(cam_name: str, ind: str, uv: np.ndarray, mask: np.ndarray, orig_wh: tuple, kernel: GaussianKernel) -> None:
    orig_w, orig_h = orig_wh
    mh, mw = mask.shape[:2]

    raw = seg_weights_from_mask(mask, uv, kernel)

    # hypothesis A: anisotropic resize directly into mask size
    uv_a = np.array([uv[0] * (mw / orig_w), uv[1] * (mh / orig_h)], dtype=float)
    aniso = seg_weights_from_mask(mask, uv_a, kernel)

    # hypothesis B: letterbox (isotropic scale + padding)
    s = min(mw / orig_w, mh / orig_h)
    pad_x = (mw - orig_w * s) / 2.0
    pad_y = (mh - orig_h * s) / 2.0
    uv_b = np.array([uv[0] * s + pad_x, uv[1] * s + pad_y], dtype=float)
    letter = seg_weights_from_mask(mask, uv_b, kernel)

    print(f"\n[{cam_name}] {ind} uv={uv}")
    print(f"  mask shape={mask.shape} orig_wh=({orig_w},{orig_h})")
    print(f"  raw:         {raw}")
    print(f"  anisotropic: {uv_a} {aniso}")
    print(f"  letterbox:   {uv_b} pad=({pad_x:.3f},{pad_y:.3f}) s={s:.6f} {letter}")


def main():
    df = pd.read_csv(DLC_CSV)
    cam_sizes = load_camera_sizes(CAMERAS_JSON)

    uvs = pick_uvs(df, FRAME, BODYPART)
    if not uvs:
        raise RuntimeError(f"No DLC rows found for frame={FRAME} bodypart='{BODYPART}'")

    kernel = GaussianKernel.make(radius=9, sigma=4.0)

    for cam_name in ["cam1_top", "cam2_front", "cam3_side"]:
        if cam_name not in uvs or len(uvs[cam_name]) == 0:
            print(f"\n[{cam_name}] No DLC uv found for frame={FRAME} bodypart='{BODYPART}'")
            continue

        viewdir = CAM_TO_VIEWDIR.get(cam_name)
        if viewdir is None:
            print(f"\n[{cam_name}] No segmentation viewdir mapping; skipping")
            continue

        mask_path = f"{SEG_ROOT}/{viewdir}/mask_full_{FRAME:06d}.png"
        mask = iio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int32)

        orig_wh = cam_sizes.get(cam_name, (960, 720))

        # Print a quick label histogram sanity-check
        vals, counts = np.unique(mask, return_counts=True)
        hist = ",".join([f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)])
        print(f"\n[{cam_name}] mask={mask_path} unique={hist}")

        for ind, uv in uvs[cam_name]:
            test_one(cam_name, ind, uv, mask, orig_wh, kernel)


if __name__ == "__main__":
    main()
