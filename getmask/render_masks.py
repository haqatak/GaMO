#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

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
# Pose loader
# ----------------------------
def load_poses(
    source_path: str,
    images_subdir: str = "coarse",
    cameras_subdir: str = "coarse",
    cameras_filename: str = "cameras.txt",
):
    src = Path(source_path)
    imgs_txt = src / "sparse" / images_subdir / "images.txt"
    cams_txt = src / "sparse" / cameras_subdir / cameras_filename

    if not imgs_txt.exists():
        raise FileNotFoundError(f"Missing images.txt: {imgs_txt}")
    if not cams_txt.exists():
        raise FileNotFoundError(f"Missing cameras file: {cams_txt}")

    cams = read_cameras_txt(cams_txt)
    imgs = read_images_txt(imgs_txt)
    cam_ids_sorted = sorted(cams.keys())

    items = []
    for name, meta in imgs.items():
        cam_id = meta["cam_id"]
        if cam_id in cams:
            entry = cams[cam_id]
        else:
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
# Find latest model directory
# ----------------------------
_TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")  # e.g., 20251022-205233

def _parse_ts(name: str):
    try:
        return datetime.strptime(name, "%Y%m%d-%H%M%S")
    except Exception:
        return None

def find_latest_model_dir(renderer_base: Path, scene_seq: str) -> Path:
    root = renderer_base / scene_seq
    if not root.exists():
        raise FileNotFoundError(f"Renderer scene root not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories under: {root}")

    ts_dirs = [(p, _parse_ts(p.name)) for p in subdirs if _TIMESTAMP_RE.match(p.name)]
    ts_dirs = [(p, ts) for p, ts in ts_dirs if ts is not None]
    if ts_dirs:
        ts_dirs.sort(key=lambda x: x[1], reverse=True)
        return ts_dirs[0][0]

    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]

# ----------------------------
# Main
# ----------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument("--source_path", required=True, help="資料根目錄（底下有 sparse/）")

    # 預設讀 coarse
    ap.add_argument("--images_subdir", default="coarse", help="讀 images.txt 的子資料夾（預設 sparse/coarse）")
    ap.add_argument("--cameras_subdir", default="coarse", help="讀 cameras 的子資料夾（預設 sparse/coarse）")
    ap.add_argument("--cameras_filename", default="cameras.txt", help="cameras 檔名（預設 cameras.txt）")
    
    # 模型來源
    ap.add_argument("--model_path", default=None, help="直接指定 EasyRenderer 模型資料夾（覆蓋自動搜尋）")
    ap.add_argument("--renderer_base", default=None, help="模型根目錄（底下有 <scene_seq>/<timestamp>）")

    # scene_seq
    ap.add_argument("--scene_seq", default=None, help="scene/seq 名稱，未指定則取 source_path 的最後一層目錄名")

    ap.add_argument("--iteration", type=int, default=10000, help="EasyRenderer 要載入的 iteration（對應你訓練好的 ckpt）")

    # ✅ 參數維持原樣
    ap.add_argument("--mask_thresh", type=float, default=0.6,
                    help="binary mask 門檻；alpha < thresh → 1，否則 0（預設 0.6）")

    ap.add_argument("--out_dir", default=None, help="mask 輸出資料夾（未指定則丟到 model_path/masks）")
    ap.add_argument("--save_rgb", action="store_true", help="同時存 render 出來的 RGB")
    ap.add_argument("--mask_mode", default="hard", choices=["soft", "binary"],
                    help="soft: 輸出連續遮罩 (1 - alpha)；binary: 以 --mask_thresh 二值化")
    
    args = ap.parse_args()

    # 解析 scene_seq
    src = Path(args.source_path).resolve()
    if args.scene_seq is not None:
        scene_seq = args.scene_seq
    else:
        if len(src.parts) >= 1:
            scene_seq = src.parts[-1]
        else:
            raise RuntimeError("無法自動推導 scene_seq，請用 --scene_seq 指定")

    # 解析 model_path
    if args.model_path is not None:
        model_path = Path(args.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"--model_path 不存在: {model_path}")
    else:
        if args.renderer_base is None:
            raise ValueError("請提供 --renderer_base 或直接指定 --model_path")
        renderer_base = Path(args.renderer_base).resolve()
        model_path = find_latest_model_dir(renderer_base, scene_seq)
    print(f"[EasyRenderer] model_path={model_path}  iteration={args.iteration}")

    # 初始化 renderer
    er = EasyRenderer(model_path=str(model_path), iteration=args.iteration)

    # 載入 poses（images.txt + cameras.txt）
    items = load_poses(
        source_path=str(src),
        images_subdir=args.images_subdir,
        cameras_subdir=args.cameras_subdir,
        cameras_filename=args.cameras_filename,
    )
    print(f"Loaded {len(items)} poses from {src}/sparse/{args.images_subdir}/images.txt")

    # 統一路徑（不分 soft/binary）
    if args.out_dir is None:
        out_dir = model_path / "masks"   # ← 統一輸出資料夾
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        (out_dir / "rgb").mkdir(exist_ok=True, parents=True)

    # 逐張渲染並存
    meta_log = []
    for (w2c_np, K_np, H, W, name) in items:
        rgb, alpha, depth = er.render(w2c_np, K_np, H, W)  # rgb:[3,H,W], alpha:[1,H,W]
        alpha = alpha.clamp(0, 1).float()
        stem = Path(name).stem

        if args.mask_mode == "soft":
            mask = (1.0 - alpha)                     # [1,H,W], 0~1
            mask_kind = "soft"
        else:
            mask = (alpha < args.mask_thresh).float()  # [1,H,W], {0,1}
            mask_kind = "binary"

        # 路徑相同；檔名仍維持 _mask.png
        mask_path = out_dir / f"{stem}_mask.png"
        torchvision.utils.save_image(mask.cpu(), str(mask_path))

        if args.save_rgb:
            rgb_path = out_dir / "rgb" / f"{stem}.png"
            torchvision.utils.save_image(rgb.clamp(0, 1).cpu(), str(rgb_path))

        meta_log.append({
            "name": name, "H": H, "W": W,
            "mask": str(mask_path),
            "type": mask_kind,
            "mask_thresh": args.mask_thresh
        })

    with open(out_dir / "masks_meta.json", "w") as f:
        json.dump(meta_log, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(items)} masks to: {out_dir}")

if __name__ == "__main__":
    main()
