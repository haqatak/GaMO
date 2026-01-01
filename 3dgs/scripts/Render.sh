#!/bin/bash

# ======== 可設定區 ========
# ROOT="Replica_6+6_perc"

# SCENE_SEQ_LIST=(
#     "office_2/Sequence_2"
#     "office_3/Sequence_1"
#     "office_4/Sequence_2"
#     "room_0/Sequence_2"
#     "room_1/Sequence_1"
#     "room_2/Sequence_1"
# )

# ROOT="Replica_crop_same_fov"
# SCENE_SEQ_LIST=(
#     "office_2_crop_0.6/Sequence_2"
#     "office_3_crop_0.6/Sequence_1"
#     "office_4_crop_0.6/Sequence_2"
#     "room_0_crop_0.6/Sequence_2"
#     "room_1_crop_0.6/Sequence_1"
#     "room_2_crop_0.6/Sequence_1"
# )

ROOT="Replica_6"

SCENE_SEQ_LIST=(
    # "office_2/"
    "office_3/"
    # "office_4/"
    # "room_0/"
    # "room_1/"
    # "room_2/"
    
)

# ROOT="Replica_6+6_perc_blend_same_fov"
# SCENE_SEQ_LIST=(
#     "office_2/Sequence_2"
#     "office_3/Sequence_1"
#     "office_4/Sequence_2"
#     "room_0/Sequence_2"
#     "room_1/Sequence_1"
#     "room_2/Sequence_1"
# )

GPU_IDS=(1)   # 多卡自動輪流分配
# ======== 結束設定區 ========


# 檢查 Python 是否存在
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please activate your environment first."
    exit 1
fi

# 逐一執行每個 {scene}/{sequence}
for i in "${!SCENE_SEQ_LIST[@]}"; do
    item="${SCENE_SEQ_LIST[$i]}"
    SCENE=$(dirname "$item")
    SEQ=$(basename "$item")
    DATA_PATH="data/Input/${ROOT}/${item}/"
    MODEL_ROOT="output/Input/${ROOT}/${item}/"

    # 找出最新的訓練結果資料夾（格式類似 20251017-043441）
    if [ -d "${MODEL_ROOT}" ]; then
        LATEST_MODEL=$(ls -td ${MODEL_ROOT}*/ 2>/dev/null | head -n 1)
        if [ -z "$LATEST_MODEL" ]; then
            echo "⚠️ No model found in ${MODEL_ROOT}, skipping..."
            continue
        fi
    else
        echo "⚠️ Directory not found: ${MODEL_ROOT}, skipping..."
        continue
    fi

    GPU_ID=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}   # 自動輪流分配 GPU
    echo "🚀 Rendering ${SCENE}/${SEQ} on GPU ${GPU_ID}"
    echo "📂 Data:  ${DATA_PATH}"
    echo "🧠 Model: ${LATEST_MODEL}"

    # 執行 render.py
    CUDA_VISIBLE_DEVICES=${GPU_ID} python render.py -s "${DATA_PATH}" -m "${LATEST_MODEL}" 

    echo "✅ Finished rendering ${SCENE}/${SEQ}"
    echo "--------------------------------------------"
done

echo "🎉 All render jobs completed!"
