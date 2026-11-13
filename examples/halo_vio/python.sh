#!/usr/bin/env bash
set -euo pipefail
TASK_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TASK_RUNTIME_DIR/runtime:/nas/shared/baidx/cuvslam_runs/cuvslam_four_vio_249244d/python_deps"
export LD_LIBRARY_PATH="/nas/shared/baidx/cuvslam_runs/cuvslam_four_vio_249244d/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3.10 "$@"
