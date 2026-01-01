#!/bin/bash
set -euo pipefail

# 用法：
# bash scripts/render_coarse_masks.sh <gpu_id> [iteration=10000] [mask_thresh=0.9] [save_rgb=0]
#
# 範例：
# bash scripts/render_coarse_masks.sh 1
# bash scripts/render_coarse_masks.sh 0 15000 0.85 1

GPU_ID=${1:-0}
ITERATION=${2:-10000}
MASK_THR=${3:-0.9}
SAVE_RGB=${4:-0}   # 1 表示也存 render 出來的 RGB

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# === 你也可以把腳本檔名改成你實際存放的新 python 檔（預設是 gen_coarse_masks.py）===
SCRIPT="gen_coarse_masks.py"

# === 資料根目錄 ===
IN_BASE="data/Input/Replica_6"

# ====== 編輯你要跑的「scene」清單（注意：這裡是 {scene} 而不是 {scene}/{sequence}） ======
scenes=(
  "room_0"       # 如果你的實際資料夾叫 room_0_crop_0.6，就填那個精確名稱
  # "room_1"
  # "room_2"
  # "office_2"
  # "office_3"
  # "office_4"
)

echo "==> GPU: ${GPU_ID}, ITER: ${ITERATION}, MASK_THR: ${MASK_THR}, SAVE_RGB: ${SAVE_RGB}"
echo "==> IN_BASE: ${IN_BASE}"
echo "==> PY: ${SCRIPT}"
echo "--------------------------------------------"

for scene in "${scenes[@]}"; do
  SCENE_ROOT="${IN_BASE}/${scene}"
  echo "==> Processing scene_root: ${SCENE_ROOT}"

  # 基本存在性檢查
  if [[ ! -d "${SCENE_ROOT}/sparse/coarse" ]]; then
    echo "⚠️  Skip: ${SCENE_ROOT}/sparse/coarse 不存在（找不到 images.txt/cameras.txt 資料夾）"
    echo "--------------------------------------------"
    continue
  fi
  if [[ ! -f "${SCENE_ROOT}/sparse/coarse/images.txt" ]]; then
    echo "⚠️  Skip: ${SCENE_ROOT}/sparse/coarse/images.txt 不存在"
    echo "--------------------------------------------"
    continue
  fi
  if [[ ! -f "${SCENE_ROOT}/sparse/coarse/cameras.txt" ]]; then
    echo "⚠️  Skip: ${SCENE_ROOT}/sparse/coarse/cameras.txt 不存在"
    echo "--------------------------------------------"
    continue
  fi

  if [[ "${SAVE_RGB}" == "1" ]]; then
    python "${SCRIPT}" \
      --scene_root "${SCENE_ROOT}" \
      --iteration "${ITERATION}" \
      --mask_thresh "${MASK_THR}" \
      --save_rgb
  else
    python "${SCRIPT}" \
      --scene_root "${SCENE_ROOT}" \
      --iteration "${ITERATION}" \
      --mask_thresh "${MASK_THR}"
  fi

  echo "✅ Done: ${scene}"
  echo "--------------------------------------------"
done

echo "🎉 All jobs completed!"
