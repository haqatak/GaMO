#!/bin/bash
set -e

# 用法：
# bash scripts/render_masks.sh <gpu_id> [iteration=10000] [mask_thresh=0.9] [save_rgb=0]
#
# 範例：
# bash scripts/render_masks.sh 1
# bash scripts/render_masks.sh 0 15000 0.85 1

GPU_ID=${1:-0}
ITERATION=${2:-10000}
MASK_THR=${3:-0.9}
SAVE_RGB=${4:-0}   # 1 表示也存render出的RGB

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ====== 編輯你要跑的資料集清單 ======
datasets=(
  'room_0_crop_0.6/Sequence_2'
  # 'room_1_crop_0.6/Sequence_1'
  # 'room_2_crop_0.6/Sequence_1'
  # 'office_2_crop_0.6/Sequence_2'
  # 'office_3_crop_0.6/Sequence_1'
  # 'office_4_crop_0.6/Sequence_2'
)

for ds in "${datasets[@]}"; do
    src="dataset/Replica_crop/${ds}"
    echo "==> Processing: ${src}"
    
    if [[ "${SAVE_RGB}" == "1" ]]; then
      python render_masks.py \
        --source_path "${src}" \
        --dataset Replica_crop \
        --iteration "${ITERATION}" \
        --images_subdir 0 \
        --cameras_subdir test \
        --cameras_filename cameras_train.txt \
        --mask_thresh "${MASK_THR}" \
        --save_rgb
    else
      python render_masks.py \
        --source_path "${src}" \
        --dataset Replica_crop \
        --iteration "${ITERATION}" \
        --images_subdir 0 \
        --cameras_subdir test \
        --cameras_filename cameras_train.txt \
        --mask_thresh "${MASK_THR}"
    fi

    echo "✅ Done: ${src}"
    echo "--------------------------------------------"
done

echo "🎉 All jobs completed!"
