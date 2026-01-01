#!/bin/bash
# =========================================================
# Pre_Process.sh
# 批次執行 pre_process.py 以生成 images_test 與 sparse/test
# =========================================================

# Python 腳本路徑
SCRIPT="scripts/pre_process.py"

# 模式與資料根目錄設定
MODE="Input"
ROOT="Replica_6"

# 要處理的場景列表
SCENES=(
    "office_2"
    "office_3"
    "office_4"
    "room_0"
    "room_1"
    "room_2"
)

# =========================================================
echo "=== Start Pre-Processing ==="
echo "Mode: ${MODE}"
echo "Root: ${ROOT}"
echo "Scenes: ${SCENES[@]}"
echo "==========================================="

# 逐一執行每個 scene
for SCENE in "${SCENES[@]}"; do
    echo ""
    echo ">>> Processing scene: ${SCENE}"
    python "${SCRIPT}" --mode "${MODE}" --root "${ROOT}" --scenes "${SCENE}"
    
    # 檢查上一個命令是否成功
    if [ $? -ne 0 ]; then
        echo "⚠️  Scene ${SCENE} failed."
    else
        echo "✅ Scene ${SCENE} done."
    fi
done

echo ""
echo "=== All scenes processed successfully! ==="
