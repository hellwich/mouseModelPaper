# Recreate the DLC TensorFlow-GPU Environment (`dlc_mouse`)

This README documents the **working setup** we arrived at for running **DeepLabCut 2.3.11** (multi-animal) with **TensorFlow 2.12.1 on GPU** in a **Conda environment only** (no system CUDA toolkit changes).

It assumes:
- NVIDIA driver is already installed and working (`nvidia-smi` works)
- You want to keep your system CUDA installation untouched
- You already have the hook scripts and setup snippets saved

---

## What this environment does

- Uses a Conda env (`dlc_mouse`) with:
  - `cudatoolkit=11.8` (env-local runtime)
  - `tensorflow==2.12.1`
  - `deeplabcut==2.3.11`
  - `tensorpack`, `tf_slim`
  - `nvidia-cudnn-cu11==8.6.0.163` (pip wheel, env-local)
- Automatically sets:
  - `PYTHONNOUSERSITE=1`
  - `LD_LIBRARY_PATH` to include env CUDA runtime and pip cuDNN libs (via activate hook)
- Supports DLC multi-animal training on GPU (confirmed by:
  - `Loaded cuDNN version 8600`
  - TensorFlow seeing GPU
  - `nvidia-smi` showing a busy python process)

---

## Important caveat (TF 2.12 vs Torch)

`tensorflow==2.12.1` pins:
- `typing_extensions < 4.6`

But your installed `torch` (used indirectly because DLC imports its PyTorch tracking module even in TF workflows) requires a newer version.

### Working compromise (confirmed)
After installing TensorFlow, run:

```bash
python -m pip install --upgrade "typing_extensions==4.15.0"
```

Pip will warn that this violates TensorFlow's declared pin; in practice, this setup **worked** for your DLC import + TF GPU + Torch import.

---

## Files to keep under version control (recommended)

Keep these together (you already saved the code snippets):

```text
dlc_env/
  environment_dlc_tf_gpu.yml        # curated env spec (optional, recommended)
  post_setup_dlc_tf_gpu.sh          # applies final typing_extensions override (optional, recommended)
  hooks/
    activate.d/dlc_mouse_env.sh     # working activate hook
    deactivate.d/dlc_mouse_env.sh   # working deactivate hook
```

If you prefer not to maintain the curated YAML yet, you can still recreate manually using the commands below.

---

## Recreate from scratch (manual commands)

### 1) Create minimal env

```bash
conda create -n dlc_mouse --override-channels -c conda-forge python=3.10 pip setuptools wheel -y
conda activate dlc_mouse
```

### 2) Install DLC + TF + required DLC TF-side dependencies

```bash
export PYTHONNOUSERSITE=1
python -m pip install deeplabcut==2.3.11
python -m pip install tensorflow==2.12.1
python -m pip install tensorpack tf_slim
```

### 3) Install env-local CUDA runtime (Conda)

```bash
conda install --override-channels -c conda-forge cudatoolkit=11.8 -y
```

### 4) Install env-local cuDNN 8.6 (pip wheel)

```bash
python -m pip install "nvidia-cudnn-cu11==8.6.0.163"
```

### 5) Fix `typing_extensions` (Torch/DLC import compatibility)

```bash
python -m pip install --upgrade "typing_extensions==4.15.0"
```

---

## Install the hooks (recommended)

These hooks make the environment reliable on every `conda activate dlc_mouse`:

- `PYTHONNOUSERSITE=1`
- `LD_LIBRARY_PATH` includes:
  - `$CONDA_PREFIX/lib` (Conda CUDA runtime)
  - pip-installed cuDNN wheel path (`nvidia.cudnn`)

### Copy hooks into the env

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp hooks/activate.d/dlc_mouse_env.sh "$CONDA_PREFIX/etc/conda/activate.d/"
cp hooks/deactivate.d/dlc_mouse_env.sh "$CONDA_PREFIX/etc/conda/deactivate.d/"
```

### Re-activate to apply hooks

```bash
conda deactivate
conda activate dlc_mouse
```

---

## Verify the environment (must pass)

### 1) Verify user-site isolation and GPU visibility in TensorFlow

```bash
python - <<'PY'
import os, site, tensorflow as tf
print("PYTHONNOUSERSITE =", os.environ.get("PYTHONNOUSERSITE"))
print("ENABLE_USER_SITE =", site.ENABLE_USER_SITE)
print("GPUs =", tf.config.list_physical_devices("GPU"))
PY
```

Expected:
- `PYTHONNOUSERSITE = 1`
- `ENABLE_USER_SITE = False`
- `GPUs = [PhysicalDevice(...GPU:0...)]`

### 2) Verify DLC + TF + Torch imports

```bash
python - <<'PY'
from importlib.metadata import version
import torch, tensorflow as tf, deeplabcut
print("typing_extensions:", version("typing_extensions"))
print("torch:", torch.__version__)
print("tensorflow:", tf.__version__)
print("deeplabcut:", deeplabcut.__version__)
PY
```

Expected (or very similar):
- `typing_extensions: 4.15.0`
- `torch: 2.10.0+cu128` (may vary)
- `tensorflow: 2.12.1`
- `deeplabcut: 2.3.11`

---

## Confirm GPU is really used during DLC training

When training starts, TensorFlow should print something like:

```text
Loaded cuDNN version 8600
```

And `nvidia-smi` should show a Python process using substantial VRAM and GPU utilization.

Example from your successful run:
- ~12.5 GB GPU memory by `python`
- ~83% GPU utilization
- `P0` performance state

---

## Typical failure modes and quick fixes

### Problem: `ImportError: cannot import name 'TypeIs' from typing_extensions`
Cause: `tensorflow==2.12.1` downgraded `typing_extensions` to 4.5.0.

Fix:
```bash
python -m pip install --upgrade "typing_extensions==4.15.0"
```

### Problem: TensorFlow sees no GPU (`[]`) and says `Cannot dlopen some GPU libraries`
Cause: missing/mismatched CUDA/cuDNN runtime libs.

Fix checklist:
- `cudatoolkit=11.8` installed in `dlc_mouse`
- `nvidia-cudnn-cu11==8.6.0.163` installed via pip
- activate hook correctly prepends `$CONDA_PREFIX/lib` and cuDNN wheel `lib` path to `LD_LIBRARY_PATH`

### Problem: DLC imports fail with weird unrelated package conflicts
Cause: user-site packages leaking into env.

Fix:
- ensure hook sets `PYTHONNOUSERSITE=1`
- verify `site.ENABLE_USER_SITE == False`

---

## Recommended inference workflow after training (for new seeds)

1. Generate new simulated behavior with a new seed (`mouse_sim2.py`)
2. Render new videos only (`mouse_deform_render_multi_swaprb.py`) — **no DLC labels needed**
3. Run DLC inference with the trained project (`predict_dlc_multi_mouse_batch.py`)
4. Use resulting H5/CSV 2D tracks later for multi-view 3D reconstruction / bundle adjustment

---

## Notes on reproducibility strategy

Prefer a **curated** `environment.yml` + hooks + post-setup script over a raw `conda env export` as your primary source of truth.

Why:
- raw exports are huge and brittle
- this environment includes an intentional TF/Torch compatibility override (`typing_extensions==4.15.0`)
- dynamic `LD_LIBRARY_PATH` logic belongs in hooks, not YAML

A full `conda env export` is still useful as an archival snapshot, but not ideal for routine recreation.

---

## Minimal curated environment example (optional)

If you later want to maintain a concise env spec, this is the right shape (not the exact hook logic):

```yaml
name: dlc_mouse
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=11.8
  - pip:
      - deeplabcut==2.3.11
      - tensorflow==2.12.1
      - tensorpack
      - tf_slim
      - nvidia-cudnn-cu11==8.6.0.163
variables:
  PYTHONNOUSERSITE: "1"
```

Then apply the hook scripts and the `typing_extensions==4.15.0` post-step.

---

## Proven-good status (from your session)

This setup has been confirmed to:
- import `torch`, `tensorflow`, `deeplabcut`
- detect the GPU in TensorFlow
- train DLC multi-animal successfully on GPU
- load cuDNN 8.6 (`Loaded cuDNN version 8600`)
- complete evaluation and skeleton selection

