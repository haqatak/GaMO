#!/usr/bin/env bash
# File: pipeline_common.sh
set -euo pipefail

# ========== Global config ==========
POINT="Duster"

# Default ROOTS / SCENES
ROOTS=("Replica_6")
declare -A SCENES
SCENES["Replica_6"]="office_2"

ENABLE_PASTE_B=${ENABLE_PASTE_B:-1}
INSET_SCALE="${INSET_SCALE:-0.6}"
FEATHER_FRAC="${FEATHER_FRAC:-0.08}"
echo "[Blend] ENABLE_PASTE_B=${ENABLE_PASTE_B} (INSET_SCALE=${INSET_SCALE}, FEATHER_FRAC=${FEATHER_FRAC})"

GPU_FIXED=${GPU_FIXED:-0}
export CUDA_VISIBLE_DEVICES="${GPU_FIXED}"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PROJ_ROOT="${PROJ_ROOT:-${SCRIPT_DIR}}"

DG_ROOT="${PROJ_ROOT}/3dgs"
MVG_ROOT="${PROJ_ROOT}/gamo"
DG_REFINE_ROOT="${PROJ_ROOT}/3dgs_refine"

echo "[Path] PROJ_ROOT is set to: ${PROJ_ROOT}"

MASK_ITER=${MASK_ITER:-10000}
FXFY_SCALE=${FXFY_SCALE:-0.6}

export CC=${CC:-/usr/bin/gcc-11}
export CXX=${CXX:-/usr/bin/g++-11}

now_ts() { date +%s; }

run_with_timer() {
  local name="$1"; shift
  local t1 t2
  echo "------------------------------"
  echo "▶▶ Start: $name"
  t1=$(now_ts)
  "$@"
  t2=$(now_ts)
  local dt=$((t2 - t1))
  echo "⏱ $name elapsed: ${dt} seconds"
  echo "------------------------------"
}

find_latest_ts_dir() {
  local base="$1"
  [[ -d "$base" ]] || { echo ""; return 0; }
  local latest
  latest=$(ls -td "${base}"/*/ 2>/dev/null | grep -E '/[0-9]{8}-[0-9]{6}/$' | head -n 1 || true)
  echo "${latest%/}"
}

views_from_root() {
  local root="$1"
  if [[ "$root" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

total_iter_for_views() {
  local v="$1"
  case "$v" in
    3)   echo 3000 ;;
    6|9) echo 7000 ;;
    *)   echo 7000 ;;
  esac
}

get_total_iter_for_root() {
  local root="$1"
  local v; v="$(views_from_root "$root")"
  local it; it="$(total_iter_for_views "${v}")"
  echo "${it}"
}

RUN_POINTCLOUD_INIT="${RUN_POINTCLOUD_INIT:-1}"
should_run_pointcloud_init() {
  [[ "${RUN_POINTCLOUD_INIT}" == "1" ]]
}

# ========== Helper: Prepare Coarse Data (Scaling fx, fy and re-indexing IDs) ==========
prepare_coarse_data() {
    local SCENE_PATH="$1"
    local SRC_DIR="${SCENE_PATH}/sparse/0"
    local DST_DIR="${SCENE_PATH}/sparse/coarse"

    echo "⚙️ Preparing coarse data in: ${DST_DIR}"
    mkdir -p "${DST_DIR}"
    cp -f "${SRC_DIR}"/*.bin "${DST_DIR}/" 2>/dev/null || true # 備份可能存在的 bin 檔
    cp -f "${SRC_DIR}"/*.txt "${DST_DIR}/"

    # 使用 Python 進行精確的文字處理
    python3 - <<'PY'
import os
from pathlib import Path

dst_dir = Path("sparse/coarse")
cam_file = dst_dir / "cameras.txt"
img_file = dst_dir / "images.txt"

# 1. 處理 cameras.txt
cam_id_map = {} # {old_id: new_id}
new_cam_lines = []
next_new_id = 700

if cam_file.exists():
    lines = cam_file.read_text().splitlines()
    for line in lines:
        if line.startswith("#") or not line.strip():
            new_cam_lines.append(line)
            continue
        
        parts = line.split()
        # 格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy ...
        old_id = parts[0]
        model = parts[1]
        
        # 建立 ID 映射
        new_id = str(next_new_id)
        cam_id_map[old_id] = new_id
        next_new_id += 1
        
        # 修改參數 (fx, fy 是 index 4, 5)
        if model in ["PINHOLE", "OPENCV", "RADIAL"]:
            parts[4] = f"{float(parts[4]) * 0.6:.12f}"
            parts[5] = f"{float(parts[5]) * 0.6:.12f}"
        
        parts[0] = new_id
        new_cam_lines.append(" ".join(parts))
    
    cam_file.write_text("\n".join(new_cam_lines) + "\n")

# 2. 處理 images.txt
if img_file.exists():
    new_img_lines = []
    lines = img_file.read_text().splitlines()
    for line in lines:
        if line.startswith("#") or not line.strip():
            new_img_lines.append(line)
            continue
        
        parts = line.split()
        # Image list 有兩行，資料行格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        if len(parts) >= 10:
            old_cid = parts[8]
            if old_cid in cam_id_map:
                parts[8] = cam_id_map[old_cid]
            new_img_lines.append(" ".join(parts))
        else:
            # 這是 POINTS2D 行，直接保留
            new_img_lines.append(line)
            
    img_file.write_text("\n".join(new_img_lines) + "\n")

print(f"✅ Coarse data processed: {len(cam_id_map)} cameras re-indexed to 700+ and scaled.")
PY
}

# ========== Step 1: 修改後的 step_initial_3dgs ==========
step_initial_3dgs() {
  local ROOT="$1"
  echo "========== (1) Initial 3DGS :: ${ROOT} =========="
  
  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"

  for SCENE in "${SCENE_LIST[@]}"; do
    local SCENE_DATA_PATH="${DG_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}"
    
    cd "${SCENE_DATA_PATH}"
    prepare_coarse_data "."
    cd "${DG_ROOT}"
    # ------------------------------------

    DATA_PATH="data/Input/${POINT}/${ROOT}/${SCENE}/"
    echo "🚀 TRAIN ${POINT}/${ROOT}/${SCENE} on GPU ${GPU_FIXED}"
    python train_o.py -s "${DATA_PATH}" --eval
  done

  # (B) render
  for SCENE in "${SCENE_LIST[@]}"; do
    DATA_PATH="data/Input/${POINT}/${ROOT}/${SCENE}/"
    MODEL_ROOT="output/${POINT}/${ROOT}/${SCENE}"
    LATEST_DIR="$(find_latest_ts_dir "${MODEL_ROOT}")"
    if [[ -z "$LATEST_DIR" ]]; then
      echo "⚠️ No latest model under: ${MODEL_ROOT}, skip rendering"
      continue
    fi
    echo "🚀 RENDER ${POINT}/${ROOT}/${SCENE} from ${LATEST_DIR} on GPU ${GPU_FIXED}"
    python render.py -s "${DATA_PATH}" -m "${LATEST_DIR}"
  done

  # (C) copy to GaMO data
  for SCENE in "${SCENE_LIST[@]}"; do
    SRC_IMAGES="${DG_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}/images"
    DST_IMAGES="${MVG_ROOT}/data/${POINT}/${ROOT}/${SCENE}/images"
    mkdir -p "$DST_IMAGES"
    rsync -av --delete "$SRC_IMAGES"/ "$DST_IMAGES"/

    MODEL_ROOT="${DG_ROOT}/output/${POINT}/${ROOT}/${SCENE}"
    LATEST_DIR="$(find_latest_ts_dir "${MODEL_ROOT}")"
    if [[ -n "$LATEST_DIR" ]]; then
      SRC_RENDERS="${LATEST_DIR}/coarse/ours_10000/renders"
      DST_RENDERS="${MVG_ROOT}/data/${POINT}/${ROOT}/${SCENE}/renders"
      mkdir -p "$DST_RENDERS"
      rsync -av --delete "$SRC_RENDERS"/ "$DST_RENDERS"/ || true
    fi
  done

  echo "========== (1) Initial 3DGS DONE :: ${ROOT} =========="
}

# ========== Step 1b: mask rendering (taming env) ==========
step_render_masks() {
  local ROOT="$1"
  echo "========== (1b) Render masks :: ${ROOT} =========="
  local GETMASK_ENTRY="${PROJ_ROOT}/getmask/render_masks.py"
  local GETMASK_OUT_BASE="${PROJ_ROOT}/getmask/output"

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"

  for SCENE in "${SCENE_LIST[@]}"; do
    SRC_PATH="${DG_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}"
    RENDERER_BASE="${DG_ROOT}/output/${POINT}/${ROOT}"
    MASK_OUT_DIR="${GETMASK_OUT_BASE}/${POINT}/${ROOT}/${SCENE}"
    DEST_MASK_DIR="${MVG_ROOT}/data/${POINT}/${ROOT}/${SCENE}/masks"
    mkdir -p "${MASK_OUT_DIR}"
    echo "🧪 Render masks for ${POINT}/${ROOT}/${SCENE} on GPU ${GPU_FIXED}"
    python "$GETMASK_ENTRY" \
      --source_path "${SRC_PATH}" \
      --renderer_base "${RENDERER_BASE}" \
      --out_dir "${MASK_OUT_DIR}" \
      --iteration "${MASK_ITER}" \
      --save_rgb
    mkdir -p "${DEST_MASK_DIR}"
    cp -f "${MASK_OUT_DIR}"/*_mask.png "${DEST_MASK_DIR}/" 2>/dev/null || true
  done

  echo "========== (1b) Render masks DONE :: ${ROOT} =========="
}

# ========== Step 2: Outpaint (GaMO env) ==========
step_outpaint() {
  local ROOT="$1"
  echo "========== (2) Outpaint :: ${ROOT} =========="
  cd "$MVG_ROOT"

  local ENTRY="run_gamo_mast3r.py"
  [[ -f "$ENTRY" ]] || { echo "❌ Cannot find $MVG_ROOT/$ENTRY"; exit 1; }

  local OUT_BASE="outputs/new/${ROOT}"
  rm -rf "${OUT_BASE}"/*
  mkdir -p "${OUT_BASE}"

  local NFRAME=${NFRAME:-80}
  local VAL_CFG=${VAL_CFG:-2.0}

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  for SCENE in "${SCENE_LIST[@]}"; do
    INPUT_PATH="data/${POINT}/${ROOT}/${SCENE}/images"
    OUTPUT_PATH="${OUT_BASE}/${SCENE}"
    mkdir -p "${OUTPUT_PATH}"
    echo "🚀 Outpaint ${ROOT}/${SCENE} on GPU ${GPU_FIXED}"
    python "$ENTRY" \
      --input_path "${INPUT_PATH}" \
      --output_path="${OUTPUT_PATH}" \
      --nframe="${NFRAME}" \
      --val_cfg="${VAL_CFG}"
  done

  echo "========== (2) Outpaint DONE :: ${ROOT} =========="
}

# ========== Step 3: Refine Align (requires numpy + pillow) ==========
step_refine_align() {
  local ROOT="$1"
  echo "========== (3) Refine Align :: ${ROOT} =========="

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  local REFINE_DST_ROOT="${DG_REFINE_ROOT}/data/Input/${POINT}/${ROOT}"
  local MVG_NEW_ROOT="${MVG_ROOT}/outputs/new/${ROOT}"
  local SCALE="${FXFY_SCALE}"

  echo "[Refine] Source: ${MVG_NEW_ROOT}"
  echo "[Refine] Dest  : ${REFINE_DST_ROOT}"
  echo "[Refine] fx,fy scale=${SCALE}"

  echo "[Step 0] Restore images/sparse from original 3dgs data"
  for SCENE in "${SCENE_LIST[@]}"; do
    SRC_3DGS_BASE="${DG_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}"
    DST_REFINE_BASE="${REFINE_DST_ROOT}/${SCENE}"
    [[ -d "${SRC_3DGS_BASE}" ]] || { echo "⚠️ No 3dgs data: ${SRC_3DGS_BASE}"; continue; }
    
    mkdir -p "${DST_REFINE_BASE}"
    
    # 同步訓練圖片 images
    if [[ -d "${SRC_3DGS_BASE}/images" ]]; then
      rsync -av --delete "${SRC_3DGS_BASE}/images/" "${DST_REFINE_BASE}/images/"
    fi
    
    # ✅ 新增：同步測試圖片 images_test (解決 FileNotFoundError)
    if [[ -d "${SRC_3DGS_BASE}/images_test" ]]; then
      echo "📦 Copying images_test for ${SCENE}..."
      rsync -av --delete "${SRC_3DGS_BASE}/images_test/" "${DST_REFINE_BASE}/images_test/"
    fi
    
    # 同步相機位姿 sparse
    if [[ -d "${SRC_3DGS_BASE}/sparse" ]]; then
      rsync -av --delete "${SRC_3DGS_BASE}/sparse/" "${DST_REFINE_BASE}/sparse/"
    fi
  done
  echo "[Step 0] Restore done."

  mapfile -t SCENES2 < <(find "${MVG_NEW_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
  if [[ ${#SCENES2[@]} -eq 0 ]]; then
    echo "[Refine] No scenes found under ${MVG_NEW_ROOT}, exit."
    return 0
  fi

  for SCENE in "${SCENES2[@]}"; do
    echo; echo "===================="
    echo "[Refine] Processing scene: ${SCENE}"
    SRC_IMG_DIR="${MVG_NEW_ROOT}/${SCENE}/images"
    DST_SCENE_DIR="${REFINE_DST_ROOT}/${SCENE}"
    DST_MVG_DIR="${DST_SCENE_DIR}/gamo"
    DST_GT_IMG_DIR="${DST_SCENE_DIR}/images"
    SPARSE_DIR="${DST_SCENE_DIR}/sparse/0"
    CAM_TXT="${SPARSE_DIR}/cameras.txt"
    IMG_TXT="${SPARSE_DIR}/images.txt"

    [[ -d "${SRC_IMG_DIR}" ]] || { echo "[Warn] No source dir: ${SRC_IMG_DIR}"; continue; }
    [[ -d "${DST_GT_IMG_DIR}" ]] || { echo "[Warn] No images dir: ${DST_GT_IMG_DIR}"; continue; }
    [[ -f "${CAM_TXT}" && -f "${IMG_TXT}" ]] || { echo "[Warn] Missing cameras.txt/images.txt under: ${SPARSE_DIR}"; continue; }

    mkdir -p "${DST_MVG_DIR}"
    rm -f "${DST_MVG_DIR}"/*.png 2>/dev/null || true

    find "${SRC_IMG_DIR}" -maxdepth 1 -type f -name "*.png" ! -name "*_ref.png" -print0 | \
      xargs -0 -I{} cp -f "{}" "${DST_MVG_DIR}/"

    # --- Align filenames ---
    DST_MVG_DIR="${DST_MVG_DIR}" DST_GT_IMG_DIR="${DST_GT_IMG_DIR}" python3 - <<'PY'
import os, sys
from pathlib import Path

mvg_dir = Path(os.environ["DST_MVG_DIR"])
img_dir = Path(os.environ["DST_GT_IMG_DIR"])

def first_digit_char_key(name: str):
    stem = Path(name).stem
    for ch in stem:
        if ch.isdigit():
            return int(ch)
    return 999

img_names = [p.name for p in img_dir.iterdir() if p.suffix.lower() == ".png"]
img_names.sort(key=first_digit_char_key)

mvg_names = [p.name for p in mvg_dir.iterdir()
             if p.suffix.lower() == ".png" and not p.name.endswith("_ref.png")]
mvg_names.sort()

n = min(len(img_names), len(mvg_names))
print(f"[Align] images={len(img_names)}, mvg={len(mvg_names)}, n={n}")
if n == 0:
    sys.exit(0)

renamed = 0
for i in range(n):
    src = mvg_dir / mvg_names[i]
    dst = mvg_dir / img_names[i]
    if src.name == dst.name:
        continue
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    renamed += 1
print(f"[Align] renamed {renamed} files")
PY

    # --- Optional blending and metadata update ---
    if [[ "${ENABLE_PASTE_B}" == "1" ]]; then
      echo "[Blend] Enabled (INSET_SCALE=${INSET_SCALE}, FEATHER_FRAC=${FEATHER_FRAC})"

      INSET_SCALE="${INSET_SCALE}" FEATHER_FRAC="${FEATHER_FRAC}" \
      DST_MVG_DIR="${DST_MVG_DIR}" DST_GT_IMG_DIR="${DST_GT_IMG_DIR}" python3 - <<'PY'
import numpy as np
from pathlib import Path
from PIL import Image
import os

mvg_dir = Path(os.environ["DST_MVG_DIR"])
img_dir = Path(os.environ["DST_GT_IMG_DIR"])
scale = float(os.environ.get("INSET_SCALE", "0.6"))
feather_frac = float(os.environ.get("FEATHER_FRAC", "0.08"))

def to_rgba(img):
    return img.convert("RGBA")

def np_from_rgba(img):
    return (np.asarray(img).astype(np.float32) / 255.0)

def rgba_from_np(arr):
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")

processed = 0
skipped = 0

pngs = sorted([p for p in mvg_dir.glob("*.png") if not p.name.endswith("_ref.png")])
for p in pngs:
    gt_path = img_dir / p.name
    if not gt_path.exists():
        print(f"[Blend] Missing GT image for {p.name}, skip")
        skipped += 1
        continue

    mvg = to_rgba(Image.open(p))
    gt = to_rgba(Image.open(gt_path))
    W, H = mvg.size

    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    if gt.size != (new_w, new_h):
        gt = gt.resize((new_w, new_h), Image.BICUBIC)

    x0 = (W - new_w) // 2
    y0 = (H - new_h) // 2
    x1 = x0 + new_w
    y1 = y0 + new_h

    base = np_from_rgba(mvg)
    patch = np_from_rgba(gt)

    patch_canvas = np.zeros_like(base, dtype=np.float32)
    patch_canvas[y0:y1, x0:x1, :] = patch

    short_side = min(W, H)
    feather_px = max(1.0, feather_frac * short_side)

    alpha = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    in_rect = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

    if np.any(in_rect):
        dx = np.minimum(xx - x0, x1 - 1 - xx)
        dy = np.minimum(yy - y0, y1 - 1 - yy)
        d = np.minimum(dx, dy)
        d = np.clip(d, 0.0, None)
        core = np.clip(d / feather_px, 0.0, 1.0)
        ramp = 0.5 - 0.5 * np.cos(np.pi * core)
        alpha[in_rect] = ramp[in_rect]

    patch_A = patch_canvas[..., 3]
    base_A = base[..., 3]

    blend_A = np.clip(alpha * patch_A + (1.0 - alpha * patch_A) * base_A, 1e-6, 1.0)
    out_rgb = (alpha[..., None] * patch_canvas[..., :3] +
               (1.0 - alpha[..., None]) * base[..., :3])
    out = np.concatenate([out_rgb, blend_A[..., None]], axis=-1)

    rgba_from_np(out).save(p)
    processed += 1

print(f"[Blend] processed={processed}, skipped={skipped}")
PY

      # copy blended images into images/ with _b suffix
      DST_MVG_DIR="${DST_MVG_DIR}" DST_GT_IMG_DIR="${DST_GT_IMG_DIR}" python3 - <<'PY'
import shutil
from pathlib import Path
import os

mvg_dir = Path(os.environ["DST_MVG_DIR"])
img_dir = Path(os.environ["DST_GT_IMG_DIR"])

count = 0
for p in sorted(mvg_dir.glob("*.png")):
    shutil.copy2(p, img_dir / f"{p.stem}_b.png")
    count += 1
print(f"[Copy] copied {count} images to images/ with _b suffix")
PY

      # update cameras.txt & images.txt
      DST_SCENE_DIR="${DST_SCENE_DIR}" DST_GT_IMG_DIR="${DST_GT_IMG_DIR}" SCALE="${SCALE}" python3 - <<'PY'
from pathlib import Path
import os

scene_dir  = Path(os.environ["DST_SCENE_DIR"])
img_dir    = Path(os.environ["DST_GT_IMG_DIR"])
sparse_dir = scene_dir / "sparse/0"
cam_txt    = sparse_dir / "cameras.txt"
img_txt    = sparse_dir / "images.txt"
scale      = float(os.environ["SCALE"])

SUPPORTED_FXFY = {"PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}
SUPPORTED_F    = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}

def read_cameras(path: Path):
    comments, entries = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            comments.append(raw)
            continue
        parts = line.split()
        entries.append((int(parts[0]), parts[1].upper(), parts[2], parts[3], parts[4:], raw))
    return comments, entries

def write_cameras(path: Path, comments, entries_new):
    bak = path.with_suffix(".txt.bak")
    if not bak.exists():
        path.replace(bak)
    else:
        path.replace(path.with_suffix(".txt.orig"))
    with path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")
        for cid, model, w, h, params in entries_new:
            f.write(f"{cid} {model} {w} {h} {' '.join(params)}\n")

def scale_params(model, params, s):
    vals = [float(x) for x in params]
    if model in SUPPORTED_FXFY and len(vals) >= 2:
        vals[0] *= s
        vals[1] *= s
    elif model in SUPPORTED_F and len(vals) >= 1:
        vals[0] *= s
    return [f"{v:.6f}" for v in vals]

lines = img_txt.read_text(encoding="utf-8").splitlines()
name2cid, meta_idx = {}, []
for i, raw in enumerate(lines):
    s = raw.strip()
    if not s or s.startswith("#"):
        continue
    parts = s.split()
    if len(parts) >= 10:
        cid = int(parts[8])
        name = parts[9]
        name2cid.setdefault(name, cid)
        meta_idx.append(i)

cam_comments, cam_entries = read_cameras(cam_txt)
cid2entry = {cid: (model, w, h, params) for cid, model, w, h, params, _ in cam_entries}

b_imgs = sorted(img_dir.glob("*_b.png"))

max_img_id = 0
for raw in lines:
    s = raw.strip()
    if not s or s.startswith("#"):
        continue
    parts = s.split()
    if len(parts) >= 10:
        try:
            max_img_id = max(max_img_id, int(parts[0]))
        except Exception:
            pass

next_img_id = max_img_id + 1
new_cam_entries, new_img_lines, added_cids = [], [], set()

def new_cid_from_old(old: int) -> int:
    return old * 100

def first_meta_line_for(name: str):
    for i in meta_idx:
        parts = lines[i].split()
        if len(parts) >= 10 and parts[9] == name:
            return parts
    return None

for bp in b_imgs:
    if not bp.name.endswith("_b.png"):
        continue
    orig_name = bp.name[:-6] + ".png"
    if orig_name not in name2cid:
        continue
    old_cid = name2cid[orig_name]
    if old_cid not in cid2entry:
        continue
    model, w, h, params = cid2entry[old_cid]
    new_cid = new_cid_from_old(old_cid)
    if new_cid not in added_cids:
        new_cam_entries.append((new_cid, model, w, h, scale_params(model, params, scale)))
        added_cids.add(new_cid)
    src_parts = first_meta_line_for(orig_name)
    if src_parts is None:
        continue
    parts = src_parts[:]
    parts[0] = str(next_img_id)
    parts[8] = str(new_cid)
    parts[9] = bp.name
    new_img_lines.append(" ".join(parts))
    new_img_lines.append("")  # empty points2d line
    next_img_id += 1

if new_cam_entries:
    existing_ids = {cid for cid, _, _, _, _, _ in cam_entries}
    merged = [(cid, model, w, h, params) for cid, model, w, h, params, _ in cam_entries]
    for rec in new_cam_entries:
        if rec[0] not in existing_ids:
            merged.append(rec)
            existing_ids.add(rec[0])
    write_cameras(cam_txt, cam_comments, merged)
    print(f"[Append] cameras.txt: +{len(new_cam_entries)} entries")

if new_img_lines:
    bak = img_txt.with_suffix(".txt.bak")
    if not bak.exists():
        img_txt.replace(bak)
    else:
        img_txt.replace(img_txt.with_suffix(".txt.orig"))
    with img_txt.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
        for l in new_img_lines:
            f.write(l + "\n")
    print(f"[Append] images.txt: +{len(new_img_lines)//2} images")
PY
    else
      echo "[Blend] Disabled: skip blending, *_b copy, and metadata append"
    fi

    echo "[Refine] Done scene: ${SCENE}"
  done

  echo "========== (3) Refine Align DONE :: ${ROOT} =========="
}

# ========== Step 3.2: Seed point cloud from 3dgs -> 3dgs_refine ==========
step_seed_pointcloud_from_3dgs() {
  local ROOT="$1"
  echo "========== (3.2) Seed point cloud :: ${ROOT} =========="

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  for SCENE in "${SCENE_LIST[@]}"; do
    local SRC_PLY="${DG_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}/sparse/0/points3D.ply"
    local DST_DIR="${DG_REFINE_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}/sparse/0"
    local DST_PLY="${DST_DIR}/points3D.ply"

    if [[ -f "${SRC_PLY}" ]]; then
      mkdir -p "${DST_DIR}"
      cp -f "${SRC_PLY}" "${DST_PLY}"
      echo "📦 Seeded: ${SRC_PLY} -> ${DST_PLY}"
    else
      echo "⚠️ Source point cloud not found: ${SRC_PLY} (skip)"
    fi
  done
  echo "========== (3.2) Seed point cloud DONE :: ${ROOT} =========="
}

# ========== Step 3.5: Pointcloud Init (dust3r) ==========
step_pointcloud_init() {
  local ROOT="$1"
  echo "========== (3.5) Pointcloud Init (dust3r) :: ${ROOT} =========="

  local DUST3R_RES_ROOT="${PROJ_ROOT}/dust3r_results/${ROOT}"
  local DG_REFINE_DATA_ROOT="${DG_REFINE_ROOT}/data/Input/${POINT}/${ROOT}"

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  for SCENE in "${SCENE_LIST[@]}"; do
    echo "🚀 pointcloud init for ${ROOT}/${SCENE} on GPU ${GPU_FIXED}"
    python "${PROJ_ROOT}/pointcloud/tools/get_replica_dust3r_pcd.py" \
      --root "${DG_REFINE_DATA_ROOT}" \
      --scenes "${SCENE}" \
      --dataset "${ROOT}" \
      --n_views 6 \
      --min_conf 1 \
      --out_root "${PROJ_ROOT}/dust3r_results"

    SRC_PLY="${DUST3R_RES_ROOT}/${SCENE}/sparse/0/points3D.ply"
    DST_PLY_DIR="${DG_REFINE_DATA_ROOT}/${SCENE}/sparse/0"
    DST_PLY="${DST_PLY_DIR}/points3D.ply"

    if [[ ! -f "${SRC_PLY}" ]]; then
      echo "⚠️ Source point cloud not found: ${SRC_PLY}"
      continue
    fi
    mkdir -p "${DST_PLY_DIR}"
    echo "📦 copy ${SRC_PLY} -> ${DST_PLY}"
    cp -f "${SRC_PLY}" "${DST_PLY}"
    echo "✅ updated: ${ROOT}/${SCENE}"
  done
  echo "========== (3.5) Pointcloud Init DONE :: ${ROOT} =========="
}

# ========== Step 4: 3dgs_refine train ==========
step_refine_train() {
  local ROOT="$1"
  echo "========== (4) 3dgs_refine train :: ${ROOT} =========="
  cd "$DG_REFINE_ROOT"

  local TRAIN_SCRIPT="${DG_REFINE_ROOT}/train.py"
  local GLOBAL_OUT_ROOT="${PROJ_ROOT}/output"
  local TOTAL_ITER
  TOTAL_ITER="$(get_total_iter_for_root "$ROOT")"
  echo "[ITER] ${ROOT} -> total_iteration=${TOTAL_ITER}"

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  for SCENE in "${SCENE_LIST[@]}"; do
    DATA_PATH="${DG_REFINE_ROOT}/data/Input/${POINT}/${ROOT}/${SCENE}"
    [[ -d "$DATA_PATH" ]] || { echo "⚠️ Data not found: $DATA_PATH"; continue; }

    echo "🚀 refine train ${ROOT}/${SCENE} on GPU ${GPU_FIXED}"
    python "$TRAIN_SCRIPT" -s "$DATA_PATH" --eval --total_iteration "${TOTAL_ITER}"

    LOCAL_OUT_DIR="${DG_REFINE_ROOT}/output/${POINT}/${ROOT}/${SCENE}"
    GLOBAL_SCENE_OUT="${GLOBAL_OUT_ROOT}/${POINT}/${ROOT}/${SCENE}"
    mkdir -p "${GLOBAL_SCENE_OUT}"
    latest_run=$(ls -1d "${LOCAL_OUT_DIR}"/*/ 2>/dev/null | sort | tail -n 1 || true)
    latest_run=${latest_run%/}
    if [[ -n "$latest_run" ]]; then
      echo "📦 Move ${latest_run} -> ${GLOBAL_SCENE_OUT}"
      rsync -av --delete "${latest_run}/" "${GLOBAL_SCENE_OUT}/$(basename "$latest_run")"/
    else
      echo "⚠️ No timestamp run found under ${LOCAL_OUT_DIR}"
    fi
  done
  echo "========== (4) 3dgs_refine train DONE :: ${ROOT} =========="
}

# ========== Step 5: 3dgs_refine render ==========
step_refine_render() {
  local ROOT="$1"
  echo "========== (5) 3dgs_refine render :: ${ROOT} =========="

  local RENDER_SCRIPT="${DG_REFINE_ROOT}/render.py"
  local OUT_ROOT="${PROJ_ROOT}/output"
  local TOTAL_ITER
  TOTAL_ITER="$(get_total_iter_for_root "$ROOT")"
  echo "[ITER] ${ROOT} -> total_iteration=${TOTAL_ITER}"

  read -a SCENE_LIST <<< "${SCENES[$ROOT]}"
  for SCENE in "${SCENE_LIST[@]}"; do
    SCENE_OUT_DIR="${OUT_ROOT}/${POINT}/${ROOT}/${SCENE}"
    if [[ ! -d "$SCENE_OUT_DIR" ]]; then
      echo "⚠️ Scene output directory not found: $SCENE_OUT_DIR"
      continue
    fi
    latest_run=$(ls -1d "$SCENE_OUT_DIR"/*/ 2>/dev/null | sort | tail -n 1 || true)
    latest_run=${latest_run%/}
    if [[ -z "$latest_run" ]]; then
      echo "⚠️ No run directory under ${SCENE_OUT_DIR}, skip"
      continue
    fi
    echo "🚀 refine render ${POINT}/${ROOT}/${SCENE} from ${latest_run} on GPU ${GPU_FIXED}"
    python "$RENDER_SCRIPT" \
      --model_path "$latest_run" \
      --iteration -1 \
      --total_iteration "${TOTAL_ITER}"
  done
  echo "========== (5) 3dgs_refine render DONE :: ${ROOT} =========="
}
