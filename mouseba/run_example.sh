#!/usr/bin/env bash
set -euo pipefail

# Run the Python example using the compiled extension.
# Usage:
#   conda activate mouseba
#   ./build.sh
#   ./run_example.sh

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX is not set. Activate your conda env first:"
  echo "  conda activate mouseba"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
PY_DIR="${ROOT_DIR}/python"

# Ensure the module exists
if ! ls "${BUILD_DIR}"/mouse_ba*.so >/dev/null 2>&1; then
  echo "ERROR: Could not find compiled extension in ${BUILD_DIR}."
  echo "Run ./build.sh first."
  exit 1
fi

export PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}"

cd "${PY_DIR}"
echo "==> Running example_run.py"
python example_run.py
