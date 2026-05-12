# --- isolate from user site-packages ---
export _OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE-__UNSET__}"
export PYTHONNOUSERSITE=1

# --- prepend conda env libs (CUDA runtime from conda) ---
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-__UNSET__}"

# Try to locate pip-installed cuDNN wheel path dynamically
_CUDNN_LIB=""
if python - <<'PY' >/dev/null 2>&1
import nvidia.cudnn
PY
then
  _CUDNN_BASE="$(python - <<'PY'
import os, nvidia.cudnn
print(os.path.dirname(nvidia.cudnn.__file__))
PY
)"
  _CUDNN_LIB="${_CUDNN_BASE}/lib"
fi

if [ -n "${_CUDNN_LIB}" ] && [ -d "${_CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

unset _CUDNN_BASE _CUDNN_LIB
