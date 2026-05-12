#!/usr/bin/env bash
set -euo pipefail

# Build the C++/pybind11 extension (Ceres backend) inside the active conda env.
# Usage:
#   conda activate mouseba
#   ./build.sh

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX is not set. Activate your conda env first:"
  echo "  conda activate mouseba"
  exit 1
fi

# Resolve project root to the directory containing this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "==> Configuring with CMake (prefix: ${CONDA_PREFIX})"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -GNinja

echo "==> Building"
cmake --build . -j

echo
echo "==> Build finished."
echo "    Extension should be in: ${BUILD_DIR}"
ls -1 "${BUILD_DIR}"/*.so 2>/dev/null || true
