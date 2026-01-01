#!/bin/bash

# ======== 可設定區 ========
#ROOT="Replica_6+6_GT_same_fov"
# ROOT="Replica_6+6_perc_blend_same_fov"

# SCENE_SEQ_LIST=(
#     "office_2/Sequence_2"
#     "office_3/Sequence_1"
#     "office_4/Sequence_2"
#     "room_0/Sequence_2"
#     "room_1/Sequence_1"
#     "room_2/Sequence_1"
    
# )

ROOT="Replica_6"

SCENE_SEQ_LIST=(
    # "office_2/"
    #"office_3/"
    #"office_4/"
    # "room_0/"
    # "room_1/"
    #"room_2/"
    
)

# 可手動設定 GPU ID，或用多張卡自動輪流分配
GPU_IDS=(1)   # 例如你有 3 張 GPU
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

    GPU_ID=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}   # 自動輪流分配 GPU
    echo "🚀 Running ${SCENE}/${SEQ} on GPU ${GPU_ID}"
    echo "📂 Path: ${DATA_PATH}"

    # 執行命令，並指定 CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train_o.py -s "${DATA_PATH}" --eval

    echo "✅ Finished ${SCENE}/${SEQ}"
    echo "--------------------------------------------"
done

echo "🎉 All jobs completed!"
