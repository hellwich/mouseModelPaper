#!/usr/bin/env bash
set -euo pipefail

# Usage: ./post_setup_dlc_tf_gpu.sh dlc_mouse
ENV_NAME="${1:-dlc_mouse}"

conda activate "$ENV_NAME"

# DLC + TF 2.12 work, but torch import via DLC needs newer typing_extensions.
python -m pip install --upgrade "typing_extensions==4.15.0"

