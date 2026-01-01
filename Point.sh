#!/usr/bin/env bash
# File: run_pointcloud_init.sh
set -euo pipefail

# 1. Path and Config Initialization
# Automatically detect project root via pipeline_common.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -f "${SCRIPT_DIR}/pipeline_common.sh" ]]; then
    source "${SCRIPT_DIR}/pipeline_common.sh"
else
    echo "❌ Cannot find pipeline_common.sh in ${SCRIPT_DIR}"
    exit 1
fi

# 2. Environment Validation Function
check_env() {
    local expected="$1"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$expected" ]]; then
        echo "❌ Environment Mismatch!"
        echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
        echo "Expected environment: $expected"
        echo "Please run: conda activate $expected"
        exit 1
    fi
}

# Ensure the correct environment is active
check_env "opamask"

# 3. Parameter Handling
# Usage: bash run_pointcloud_init.sh [ROOT] [SCENE]
# Example: bash run_pointcloud_init.sh Replica_6 office_2
TARGET_ROOT="${1:-Replica_6}"
TARGET_SCENE="${2:-office_2}"

DATA_ROOT="${DG_ROOT}/data/Input/${POINT}" 
PY_ENTRY="${PROJ_ROOT}/pointcloud/tools/get_replica_dust3r_pcd.py"

N_VIEWS=6
MIN_CONF=1

# ==============================================
echo "PROJ_ROOT    : ${PROJ_ROOT}"
echo "Target Pair  : ${TARGET_ROOT} / ${TARGET_SCENE}"
echo "N_VIEWS      : ${N_VIEWS}"
echo "MIN_CONF     : ${MIN_CONF}"
echo "GPU_VISIBLE  : ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================="

# 4. Execution
# Check if the data path exists before running
SCENE_PATH="${DATA_ROOT}/${TARGET_ROOT}/${TARGET_SCENE}"

if [[ ! -d "$SCENE_PATH" ]]; then
    echo "❌ Error: Scene path does not exist: $SCENE_PATH"
    exit 1
fi

echo "🚀 Starting Pointcloud Init for ${TARGET_ROOT}/${TARGET_SCENE}..."

python "${PY_ENTRY}" \
  --root "${DATA_ROOT}/${TARGET_ROOT}" \
  --scenes "${TARGET_SCENE}" \
  --dataset "${TARGET_ROOT}" \
  --n_views "${N_VIEWS}" \
  --min_conf "${MIN_CONF}" \
  --out_root "${PROJ_ROOT}/dust3r_results"

echo "---------------------------------------------"
echo "✅ Process finished for ${TARGET_ROOT}/${TARGET_SCENE}"