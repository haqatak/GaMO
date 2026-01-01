#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./eval.sh <GPU_ID>
# Example:
#   ./eval.sh 0

export CUDA_VISIBLE_DEVICES="${1:?GPU_ID is required}"

# 要跑的場景
datasets=('office_2/Sequence_2')
Root="Replica_6+6_perc_blend_same_fov"   # 這裡用字串就好，不要用陣列

for ds in "${datasets[@]}"; do
  SRC_PATH="data/${Root}/${ds}"
  BASE_WORK="output/${Root}/${ds}"

  echo "Processing dataset:"
  echo "  SOURCE : ${SRC_PATH}"
  echo "  WORK   : ${BASE_WORK}"

  # 1) 先嘗試直接用 BASE_WORK（若其中已經直接有 cfg_args）
  MODEL_PATH="${BASE_WORK}"
  if [[ ! -f "${MODEL_PATH}/cfg_args" ]]; then
    # 2) 沒有 cfg_args -> 從 BASE_WORK 底下挑「最新的日期資料夾」
    #    例如 output/.../office_4/Sequence_2/20251019-015317
    latest_dir="$(ls -1dt "${BASE_WORK}"/*/ 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_dir}" ]]; then
      MODEL_PATH="${latest_dir%/}"
    fi
  fi

  if [[ ! -f "${MODEL_PATH}/cfg_args" ]]; then
    echo "❌ 找不到 cfg_args：${BASE_WORK} 或其子資料夾底下都沒有 cfg_args"
    echo "   請確認訓練輸出結構正確。"
    exit 1
  fi

  echo "  MODEL  : ${MODEL_PATH}"

  # 渲染
  python render_tam.py \
    --source_path "${SRC_PATH}" \
    --model_path  "${MODEL_PATH}" \
    --iteration   10000

  # 指標
  python metrics_tam.py \
    --source_path "${SRC_PATH}" \
    --model_path  "${MODEL_PATH}" \
    --iteration   10000
done

echo "✅ ALL DONE."
