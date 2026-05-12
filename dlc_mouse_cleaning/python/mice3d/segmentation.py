from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

import numpy as np


@dataclass
class GaussianKernel:
    radius: int
    sigma: float
    weights: np.ndarray  # (2r+1, 2r+1), sum=1

    @staticmethod
    def make(radius: int, sigma: float) -> "GaussianKernel":
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        r = radius
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        w = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        w_sum = float(w.sum())
        if w_sum <= 0:
            w = np.zeros_like(w)
        else:
            w = w / w_sum
        return GaussianKernel(radius=radius, sigma=sigma, weights=w.astype(float))


class SegmentationSource:
    """Abstract segmentation source.

    Must return a 2D integer mask where:
      0 = background
      1 = mouse0 segment
      2 = mouse1 segment

    If your segmentation uses different labels, map them before passing.
    """

    def get_mask(self, cam_name: str, frame: int) -> np.ndarray:
        raise NotImplementedError


class SegmentationVideoSource(SegmentationSource):
    """Reads per-camera segmentation masks from videos.

    Assumes each frame is either:
      - single-channel labels 0/1/2, or
      - RGB where labels are encoded as grayscale values 0/1/2.

    You can provide a custom `decode_fn(frame_bgr)->mask` if needed.
    """

    def __init__(self, video_paths: Dict[str, str], decode_fn=None, backend: str = "imageio"):
        backend = str(backend)
        self._backend = backend
        self._decode_fn = decode_fn
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

        if self._backend == "opencv":
            import cv2
            self._cv2 = cv2
            self._caps = {cam: cv2.VideoCapture(path) for cam, path in video_paths.items()}
        else:
            import imageio.v3 as iio
            self._iio = iio
            self._video_paths = dict(video_paths)

    def get_mask(self, cam_name: str, frame: int) -> np.ndarray:
        key = (cam_name, frame)
        if key in self._cache:
            return self._cache[key]

        if self._backend == "opencv":
            cap = self._caps[cam_name]
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read segmentation frame {frame} for {cam_name}")
        else:
            path = self._video_paths[cam_name]
            try:
                img = self._iio.imread(path, index=frame)
            except Exception as e:
                raise RuntimeError(f"Cannot read segmentation frame {frame} for {cam_name} from {path}: {e}")

        if self._decode_fn is not None:
            mask = self._decode_fn(img)
        else:
            if img.ndim == 3:
                # BGR -> gray
                mask = img[:, :, 0].astype(np.int32)
            else:
                mask = img.astype(np.int32)

        self._cache[key] = mask
        # simple cache policy: keep last few
        if len(self._cache) > 12:
            # pop arbitrary oldest (dict insertion order in py3.7+)
            for k in list(self._cache.keys())[:6]:
                self._cache.pop(k, None)
        return mask


class SegmentationDirectorySource(SegmentationSource):
    """Reads masks from a directory per camera.

    Expected layout:
      root/<cam_name>/frame_000000.png

    This class also supports *auto-detection* of common alternate conventions, e.g.:
      root/top/mask_000001.png
      root/front/mask_000001.png
      root/side/mask_000001.png

    Auto-detection uses:
      - camera subdirectory matching ("cam1_top" -> "cam1_top" or "top")
      - filename pattern inference for prefix (frame/mask/...), padding, and start index

    You can override any of these via constructor arguments.

    Labels must be 0/1/2.
    """

    def __init__(
        self,
        root_dir: str,
        ext: Optional[str] = None,
        backend: str = "imageio",
        *,
        cam_names: Optional[Iterable[str]] = None,
        cam_dir_map: Optional[Dict[str, str]] = None,
        filename_prefix: Optional[str] = None,
        pad: Optional[int] = None,
        start_index: Optional[int] = None,
        strict: bool = True,
    ):
        backend = str(backend)
        self._backend = backend
        import os
        import re
        self._os = os
        self._re = re
        self.root_dir = root_dir
        self.ext = ext
        self._strict = bool(strict)

        self._cam_names = list(cam_names) if cam_names is not None else None
        self._cam_dir_map_user = dict(cam_dir_map) if cam_dir_map else {}
        self._filename_prefix_user = filename_prefix
        self._pad_user = pad
        self._start_index_user = start_index

        # caches
        self._cam_dir_cache: Dict[str, str] = {}
        # per cam_dir: (prefix, pad, ext, start_index)
        self._pattern_cache: Dict[str, Tuple[str, int, str, int]] = {}

        if self._backend == "opencv":
            import cv2
            self._cv2 = cv2
        else:
            import imageio.v3 as iio
            self._iio = iio

    def _list_subdirs(self) -> Dict[str, str]:
        """Return map lower_name->actual_name for immediate subdirs."""
        sub: Dict[str, str] = {}
        try:
            for name in self._os.listdir(self.root_dir):
                p = self._os.path.join(self.root_dir, name)
                if self._os.path.isdir(p):
                    sub[name.lower()] = name
        except Exception:
            pass
        return sub

    def _resolve_cam_dir(self, cam_name: str) -> str:
        """Map a camera name (e.g. cam1_top) to a subdirectory (e.g. top)."""
        if cam_name in self._cam_dir_cache:
            return self._cam_dir_cache[cam_name]

        # explicit mapping wins
        if cam_name in self._cam_dir_map_user:
            d = self._cam_dir_map_user[cam_name]
            self._cam_dir_cache[cam_name] = d
            return d

        subdirs = self._list_subdirs()
        if not subdirs:
            # allow root itself if no subdirs
            self._cam_dir_cache[cam_name] = ""
            return ""

        # generate candidate tokens
        base = cam_name
        suffix = self._re.sub(r"^cam\d+_", "", base)
        tokens = {base.lower(), suffix.lower()}
        tokens.update([t.lower() for t in base.split("_") if t])
        tokens.update([t.lower() for t in suffix.split("_") if t])

        best = None
        best_score = -1
        for sub_l, sub_actual in subdirs.items():
            score = 0
            if sub_l == base.lower():
                score = 100
            elif sub_l == suffix.lower():
                score = 90
            else:
                for tok in tokens:
                    if sub_l == tok:
                        score = max(score, 80)
                    elif tok and (tok in sub_l or sub_l in tok):
                        score = max(score, 50)
            if score > best_score:
                best_score = score
                best = sub_actual

        if best is None:
            msg = f"Cannot map camera '{cam_name}' to a subdirectory under '{self.root_dir}'. Found: {list(subdirs.values())}"
            if self._strict:
                raise RuntimeError(msg)
            best = cam_name

        self._cam_dir_cache[cam_name] = best
        return best

    def _infer_pattern_for_dir(self, cam_dir: str) -> Tuple[str, int, str, int]:
        """Infer (prefix, pad, ext, start_index) from filenames in cam_dir."""
        if cam_dir in self._pattern_cache:
            return self._pattern_cache[cam_dir]

        # user overrides
        prefix = self._filename_prefix_user
        pad = self._pad_user
        start_index = self._start_index_user
        ext = self.ext

        dir_path = self._os.path.join(self.root_dir, cam_dir) if cam_dir else self.root_dir
        try:
            files = [f for f in self._os.listdir(dir_path) if self._os.path.isfile(self._os.path.join(dir_path, f))]
        except Exception as e:
            raise RuntimeError(f"Cannot list segmentation directory: {dir_path} ({e})")

        # If ext not specified, infer from common image extensions
        if ext is None:
            exts: Dict[str, int] = {}
            for f in files:
                m = self._re.match(r"^.+\.(png|jpg|jpeg|tif|tiff)$", f, flags=self._re.IGNORECASE)
                if m:
                    ex = m.group(1).lower()
                    exts[ex] = exts.get(ex, 0) + 1
            ext = sorted(exts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if exts else "png"

        if prefix is not None and pad is not None and start_index is not None:
            self._pattern_cache[cam_dir] = (str(prefix), int(pad), str(ext), int(start_index))
            return self._pattern_cache[cam_dir]

        rx = self._re.compile(r"^(?P<prefix>[A-Za-z0-9]+)_(?P<idx>\d+)\.(?P<ext>\w+)$")
        stats: Dict[Tuple[str, str, int], list[int]] = {}
        for f in files[:5000]:
            m = rx.match(f)
            if not m:
                continue
            pr = m.group("prefix")
            ix = int(m.group("idx"))
            ex = m.group("ext").lower()
            key = (pr, ex, len(m.group("idx")))
            stats.setdefault(key, []).append(ix)

        if not stats:
            pr = prefix or "frame"
            pd = pad or 6
            st = start_index or 0
            self._pattern_cache[cam_dir] = (str(pr), int(pd), str(ext), int(st))
            return self._pattern_cache[cam_dir]

        def score_item(kv):
            (pr, ex, pd), idxs = kv
            cnt = len(idxs)
            bonus = 0
            if prefix is not None and pr == prefix:
                bonus += 1000
            if pr.lower() in ("mask", "frame"):
                bonus += 10
            if ex == str(ext).lower():
                bonus += 5
            return (cnt, bonus)

        (pr, ex, pd), idxs = sorted(stats.items(), key=score_item, reverse=True)[0]
        if prefix is None:
            prefix = pr
        if pad is None:
            pad = pd
        if start_index is None:
            start_index = min(idxs)
        ext = ex

        self._pattern_cache[cam_dir] = (str(prefix), int(pad), str(ext), int(start_index))
        return self._pattern_cache[cam_dir]

    def _candidate_paths(self, cam_dir: str, frame: int) -> list[str]:
        prefix, pad, ext, start_idx = self._infer_pattern_for_dir(cam_dir)
        dir_path = self._os.path.join(self.root_dir, cam_dir) if cam_dir else self.root_dir

        idx = frame + start_idx
        paths = [self._os.path.join(dir_path, f"{prefix}_{idx:0{pad}d}.{ext}")]

        for alt_prefix in ["mask", "frame"]:
            if alt_prefix != prefix:
                paths.append(self._os.path.join(dir_path, f"{alt_prefix}_{idx:0{pad}d}.{ext}"))

        paths.append(self._os.path.join(dir_path, f"{prefix}_{frame:0{pad}d}.{ext}"))
        paths.append(self._os.path.join(dir_path, f"{prefix}_{frame+1:0{pad}d}.{ext}"))
        if frame > 0:
            paths.append(self._os.path.join(dir_path, f"{prefix}_{frame-1:0{pad}d}.{ext}"))

        out = []
        seen = set()
        for p in paths:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def get_mask(self, cam_name: str, frame: int) -> np.ndarray:
        cam_dir = self._resolve_cam_dir(cam_name)
        paths = self._candidate_paths(cam_dir, frame)

        path = None
        for p in paths:
            if self._os.path.exists(p):
                path = p
                break
        if path is None:
            raise RuntimeError(
                f"Cannot find segmentation mask for cam='{cam_name}', frame={frame}. Tried: {paths[:4]}{'...' if len(paths)>4 else ''}"
            )
        if self._backend == "opencv":
            img = self._cv2.imread(path, self._cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot read mask: {path}")
        else:
            try:
                img = self._iio.imread(path)
            except Exception as e:
                raise RuntimeError(f"Cannot read mask: {path} ({e})")
        if img.ndim == 3:
            img = img[:, :, 0]
        return img.astype(np.int32)


def seg_weights_from_mask(mask: np.ndarray, uv: np.ndarray, kernel: GaussianKernel) -> Tuple[float, float, float]:
    """Compute (w_bg,w_m0,w_m1) around uv using weighted counts."""
    h, w = mask.shape[:2]
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))

    r = kernel.radius
    x0 = u - r
    y0 = v - r
    x1 = u + r
    y1 = v + r

    # clip
    xx0 = max(0, x0)
    yy0 = max(0, y0)
    xx1 = min(w - 1, x1)
    yy1 = min(h - 1, y1)

    patch = mask[yy0 : yy1 + 1, xx0 : xx1 + 1]

    # matching kernel window
    kx0 = xx0 - x0
    ky0 = yy0 - y0
    kx1 = kx0 + (xx1 - xx0)
    ky1 = ky0 + (yy1 - yy0)
    kw = kernel.weights[ky0 : ky1 + 1, kx0 : kx1 + 1]

    # accumulate
    w_bg = float(kw[patch == 0].sum())
    w_m0 = float(kw[patch == 1].sum())
    w_m1 = float(kw[patch == 2].sum())

    s = w_bg + w_m0 + w_m1
    if s <= 0:
        return (0.0, 0.0, 0.0)
    return (w_bg / s, w_m0 / s, w_m1 / s)
