#!/usr/bin/env bash
set -euo pipefail
TASK_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TASK_RUNTIME_DIR/runtime:$TASK_RUNTIME_DIR/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TASK_RUNTIME_DIR/runtime/cuvslam:$TASK_RUNTIME_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "${CUVSLAM_PYTHON:-python3.10}" "$@"
