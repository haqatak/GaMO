#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser

from utils.easy_renderer import EasyRenderer   # 不改 easy_renderer
import torchvision

# ----------------------------
# COLMAP helpers
# ----------------------------
def read_cameras_txt(path: Path):
    cams = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            model  = toks[1].upper()
            width  = int(toks[2])
            height = int(toks[3])
            params = np.array(list(map(float, toks[4:])), dtype=np.float64)
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cams

def read_images_txt(path: Path):
    imgs = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            i += 1
            continue
        toks = line.split()
        # IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
        q = np.array(list(map(float, toks[1:5])), dtype=np.float64)   # qw qx qy qz
        t = np.array(list(map(float, toks[5:8])), dtype=np.float64)   # tx ty tz
        cam_id = int(toks[8])
        name   = " ".join(toks[9:])
        imgs[name] = {"qvec": q, "tvec": t, "cam_id": cam_id}
        i += 2  # skip 2D points line
    return imgs

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

def intrinsics_from_entry(entry):
    model = entry["model"]
    w, h = entry["width"], entry["height"]
    p = entry["params"]
    if model in ("PINHOLE","OPENCV","OPENCV_FISHEYE","BROWN_CONRADY","FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("SIMPLE_PINHOLE","SIMPLE_RADIAL","RADIAL"):
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    return K.astype(np.float32), int(h), int(w)

# ----------------------------
# Pose loader: 強制從 sparse/coarse 讀 images.txt + cameras.txt
# ----------------------------
def load_poses_from_coarse(scene_root: Path):
    imgs_txt = scene_root / "sparse" / "coarse" / "images.txt"
    cams_txt = scene_root / "sparse" / "coarse" / "cameras.txt"

    if not imgs_txt.exists():
        raise FileNotFoundError(f"Missing images.txt: {imgs_txt}")
    if not cams_txt.exists():
        raise FileNotFoundError(f"Missing cameras.txt: {cams_txt}")

    cams = read_cameras_txt(cams_txt)
    imgs = read_images_txt(imgs_txt)
    cam_ids_sorted = sorted(cams.keys())

    items = []
    for name, meta in imgs.items():
        cam_id = meta["cam_id"]
        if cam_id in cams:
            entry = cams[cam_id]
        else:
            # 容錯：單一鏡頭或 index 對齊
            if len(cam_ids_sorted) == 1:
                entry = cams[cam_ids_sorted[0]]
            elif 0 <= cam_id < len(cam_ids_sorted):
                entry = cams[cam_ids_sorted[cam_id]]
            else:
                raise KeyError(f"CAMERA_ID {cam_id} not in {cam_ids_sorted}")

        K, H, W = intrinsics_from_entry(entry)

        R = qvec2rotmat(meta["qvec"])  # world->camera
        t = meta["tvec"].reshape(3, 1)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3]  = t[:, 0].astype(np.float32)

        items.append((w2c, K, H, W, name))
    return items

# ----------------------------
# Utilities
# ----------------------------
def _find_scene_name(scene_root: Path) -> str:
    """從 data/Input/Replica_6/{scene} 取出 {scene} 名字。"""
    # 優先用最後一層資料夾名
    return scene_root.name

def _latest_subdir(root: Path) -> Path:
    """在 root 下找最後修改時間最新的子資料夾。若沒有子資料夾則回傳 root 自己。"""
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        return root
    return max(subdirs, key=lambda d: d.stat().st_mtime)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument(
        "--scene_root",
        required=True,
        help="指到 data/Input/Replica_6/{scene} 這一層目錄",
    )
    ap.add_argument(
        "--iteration",
        type=int,
        default=10000,
        help="EasyRenderer 要載入的 iteration（對應你訓練好的 ckpt）",
    )
    ap.add_argument(
        "--mask_thresh",
        type=float,
        default=0.9,
        help="（保留參數，不更動 easy_renderer；目前輸出 soft mask = 1 - alpha）",
    )
    ap.add_argument(
        "--save_rgb",
        action="store_true",
        help="同時儲存渲染出的 RGB 到同一路徑底下的 rgb/ 資料夾",
    )
    args = ap.parse_args()

    scene_root = Path(args.scene_root).resolve()
    if not scene_root.exists():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")

    # 1) 從 sparse/coarse 讀 poses
    items = load_poses_from_coarse(scene_root)
    print(f"[Coarse] Loaded {len(items)} poses from {scene_root/'sparse'/'coarse'}")

    # 2) 找到 output/Input/Replica_6/{scene}/{最新時間}
    scene_name = _find_scene_name(scene_root)
    model_base = Path("output") / "Input" / "Replica_6" / scene_name
    latest_root = _latest_subdir(model_base)
    if latest_root == model_base:
        # 若沒有子資料夾就視為本身是訓練結果目錄
        print(f"[Info] No timestamped subdir found under {model_base}, use itself as model root.")
    print(f"[Model Root] {latest_root}")

    # 3) 初始化 EasyRenderer（使用最新時間資料夾作為 model_path）
    print(f"[EasyRenderer] model_path={latest_root}  iteration={args.iteration}")
    er = EasyRenderer(model_path=str(latest_root), iteration=args.iteration)

    # 4) 設定輸出資料夾：output/Input/Replica_6/{scene}/{最新時間}/coarse/masks
    out_dir = latest_root / "coarse" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        (out_dir / "rgb").mkdir(parents=True, exist_ok=True)

    # 5) 逐張渲染並存 soft mask（1 - alpha）
    meta_log = []
    for (w2c_np, K_np, H, W, name) in items:
        # 丟 numpy 給 easy_renderer
        rgb, alpha, depth = er.render(w2c_np, K_np, H, W)  # rgb:[3,H,W], alpha:[1,H,W] (torch)

        # soft mask = 1 - alpha
        soft_mask = (1.0 - alpha.clamp(0, 1))

        stem = Path(name).stem
        mask_path = out_dir / f"{stem}_mask.png"
        torchvision.utils.save_image(soft_mask.cpu(), str(mask_path))

        if args.save_rgb:
            rgb_path = out_dir / "rgb" / f"{stem}.png"
            torchvision.utils.save_image(rgb.clamp(0, 1).cpu(), str(rgb_path))

        meta_log.append({
            "name": name,
            "H": H,
            "W": W,
            "mask": str(mask_path.relative_to(latest_root)),  # 記錄相對於最新根目錄
        })

    with open(out_dir / "masks_meta.json", "w") as f:
        json.dump(meta_log, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(items)} masks to: {out_dir}")

if __name__ == "__main__":
    main()
