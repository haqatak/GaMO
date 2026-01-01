#!/usr/bin/env bash
# File: Pipeline.sh
set -euo pipefail

# Load common configurations and functions
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"

show_usage() {
    echo "Usage: bash Pipeline.sh --step [1|1b|2|3|3.5|4|5] [ROOT_NAME] [SCENE_NAME]"
    echo ""
    echo "Example: bash Pipeline.sh --step 1 Replica_6 office_2"
    echo ""
    echo "Step to Environment Mapping:"
    echo "  1  : Initial 3DGS Training/Rendering (Env: 3dgs)"
    echo "  1b : Mask Generation                (Env: opamask)"
    echo "  2  : GaMO Outpainting               (Env: GaMO)"
    echo "  3  : Alignment & Pointcloud Init    (Env: GaMO)"
    echo "  3.5: Pointcloud Init (Dust3R)       (Env: opamask)"
    echo "  4  : Refine Training                (Env: 3dgs)"
    echo "  5  : Refine Rendering               (Env: 3dgs)"
}

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

STEP=""
ROOT_NAME=""
SCENE_NAME=""

# 支援原本的 --step 1 --root Replica_6 
# 也支援新要求的 bash Pipeline.sh --step 1 Replica_6 office_2
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            # 檢查接下來的參數是否為 ROOT 和 SCENE (不是以 -- 開頭的)
            if [[ $# -gt 0 && ! $1 == --* ]]; then
                ROOT_NAME="$1"; shift
            fi
            if [[ $# -gt 0 && ! $1 == --* ]]; then
                SCENE_NAME="$1"; shift
            fi
            ;;
        --root)
            ROOT_NAME="$2"
            shift 2
            ;;
        *)
            show_usage; exit 1
            ;;
    esac
done

if [[ -z "$STEP" || -z "$ROOT_NAME" ]]; then 
    echo "❌ Error: Missing Step or Root Name."
    show_usage; exit 1; 
fi

# 如果沒有指定 SCENE_NAME，後面 common.sh 的 get_scene_list 會自動處理 fallback
echo "📍 Target: Root=${ROOT_NAME}, Scene=${SCENE_NAME:-ALL_IN_CONFIG}"

case "$STEP" in
    1)
        check_env "3dgs"
        run_with_timer "Step 1: Initial 3DGS" step_initial_3dgs "$ROOT_NAME" "$SCENE_NAME"
        ;;
    1b)
        check_env "opamask"
        run_with_timer "Step 1b: Render Masks" step_render_masks "$ROOT_NAME" "$SCENE_NAME"
        ;;
    2)
        check_env "GaMO"
        run_with_timer "Step 2: Outpaint" step_outpaint "$ROOT_NAME" "$SCENE_NAME"
        ;;
    3)
        check_env "GaMO"
        # 這裡需要把 SCENE_NAME 也傳進去子 shell
        run_with_timer "Step 3a: Refine Align & Seed" \
            bash -c "source $(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh && step_refine_align $ROOT_NAME $SCENE_NAME && step_seed_pointcloud_from_3dgs $ROOT_NAME $SCENE_NAME"
        
        if should_run_pointcloud_init; then
            echo "-------------------------------------------------------"
            echo "👉 Part A (Alignment) finished."
            echo "👉 Part B (Pointcloud Init) requires 'opamask' environment."
            echo "Please run: conda activate opamask && bash Pipeline.sh --step 3.5 $ROOT_NAME $SCENE_NAME"
            echo "-------------------------------------------------------"
        fi
        ;;
    3.5)
        check_env "opamask"
        run_with_timer "Step 3.5: Pointcloud Init (Dust3R)" step_pointcloud_init "$ROOT_NAME" "$SCENE_NAME"
        ;;
    4)
        check_env "3dgs"
        run_with_timer "Step 4: Refine Train" step_refine_train "$ROOT_NAME" "$SCENE_NAME"
        ;;
    5)
        check_env "3dgs"
        run_with_timer "Step 5: Refine Render" step_refine_render "$ROOT_NAME" "$SCENE_NAME"
        ;;
    *)
        echo "Invalid step: $STEP"
        show_usage
        exit 1
        ;;
esac