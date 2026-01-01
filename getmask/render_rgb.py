#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

from utils.easy_renderer_alpha import EasyRenderer   # 不改 easy_renderer
import torchvision

# 嘗試匯入 imageio 做 mp4
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("[Warn] imageio 未安裝，將只輸出 PNG，不產生 mp4（可使用 pip install imageio[ffmpeg]）")

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
            cams[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
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

    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "BROWN_CONRADY", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx = p[1]
        cy = p[2]
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
    images_subdir: str = "test",
    cameras_subdir: str = "test",
    cameras_filename: str = "cameras.txt",
):
    """
    從 source_path/sparse/<images_subdir>/images.txt
    和 source_path/sparse/<cameras_subdir>/<cameras_filename>
    讀取 COLMAP pose + intrinsics。
    """
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
            # 保險處理：如果 images 裡的 CAMERA_ID 跟 cameras.txt 不 match
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
# Find largb model directory
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

# ===== axis-angle 旋轉（在世界座標繞某個 axis 轉） =====
def axis_angle_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c
    R = np.array([
        [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float64)
    return R

# ----------------------------
# Main
# ----------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument("--source_path", required=True,
                    help="資料根目錄（底下有 sparse/，預設讀 sparse/test）")

    # 預設讀 test
    ap.add_argument("--images_subdir", default="test",
                    help="讀 images.txt 的子資料夾（預設 sparse/test）")
    ap.add_argument("--cameras_subdir", default="test",
                    help="讀 cameras 的子資料夾（預設 sparse/test）")
    ap.add_argument("--cameras_filename", default="cameras.txt",
                    help="cameras 檔名（預設 cameras.txt）")
    
    # 模型來源（GraphDECO / EasyRenderer 模型）
    ap.add_argument("--model_path", default=None,
                    help="直接指定 EasyRenderer 模型資料夾（覆蓋自動搜尋）")
    ap.add_argument("--renderer_base", default=None,
                    help="模型根目錄（底下有 <scene_seq>/<timestamp>）")

    # scene_seq
    ap.add_argument("--scene_seq", default=None,
                    help="scene/seq 名稱，未指定則取 source_path 的最後一層目錄名")

    ap.add_argument("--iteration", type=int, default=10000,
                    help="EasyRenderer 要載入的 iteration（對應你訓練好的 ckpt）")

    # RGB 輸出資料夾
    ap.add_argument("--out_dir", default=None,
                    help="render RGB 輸出資料夾（未指定則丟到 model_path/test_pack）")

    # mp4 fps
    ap.add_argument("--fps", type=int, default=30,
                    help="輸出 mp4 的 FPS（預設 30）")

    # 總 frame 數量（包含頭尾）
    ap.add_argument("--num_frames", type=int, default=150,
                    help="整段影片要輸出的總 frame 數（預設 150）")

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
    print(f"[EasyRenderer RGB] model_path={model_path}  iteration={args.iteration}")

    # 初始化 renderer
    er = EasyRenderer(model_path=str(model_path), iteration=args.iteration)

    # 載入 poses（images.txt + cameras.txt），預設 sparse/test
    items = load_poses(
        source_path=str(src),
        images_subdir=args.images_subdir,
        cameras_subdir=args.cameras_subdir,
        cameras_filename=args.cameras_filename,
    )
    print(f"Loaded {len(items)} poses from {src}/sparse/{args.images_subdir}/images.txt")

    if len(items) < 1:
        raise RuntimeError("至少需要一個 test view")

    # 依照檔名排序，取「中間那一張」的方向當 base
    items_sorted = sorted(items, key=lambda x: x[-1])
    mid_idx = len(items_sorted) // 2
    w2c_base, K_base, H_base, W_base, name_base = items_sorted[mid_idx]
    print(f"Base view for 360° spin (orientation): {name_base}")

    # intrinsics / resolution 用 base view，並把 FOV 拉廣 (fx, fy x 0.8)
    K = K_base.copy().astype(np.float32)
    K[0, 0] *= 0.8  # fx
    K[1, 1] *= 0.8  # fy
    H, W = H_base, W_base
    print("Scaled intrinsics: fx, fy = ",
          float(K[0, 0]), float(K[1, 1]))

    # ---------- 所有 camera center + bbox 中心 ----------
    centers = []
    up_vecs = []   # 每個相機的「up 向量」（世界座標）
    for (w2c, _, _, _, _) in items:
        R_i = w2c[:3, :3].astype(np.float64)   # world -> camera
        t_i = w2c[:3, 3].astype(np.float64)
        C_i = -R_i.T @ t_i                     # camera center in world

        R_c2w_i = R_i.T                        # camera -> world
        # COLMAP 的 camera y 軸是「往下」，所以取 -y 當「up」
        up_i = -R_c2w_i[:, 1]

        centers.append(C_i)
        up_vecs.append(up_i)

    centers = np.stack(centers, axis=0)
    up_vecs = np.stack(up_vecs, axis=0)

    # bbox 幾何中心
    min_xyz = centers.min(axis=0)
    max_xyz = centers.max(axis=0)
    C_bbox = 0.5 * (min_xyz + max_xyz)

    # y（高度）用平均，避免太高/太低 outlier
    C = C_bbox.copy()
    C[1] = centers[:, 1].mean()

    print("BBox center (world):", C_bbox)
    print("Final spin center (world):", C)

    # ---------- 全域 up 向量（估計世界的「正上方」） ----------
    global_up = up_vecs.mean(axis=0)
    global_up = global_up / (np.linalg.norm(global_up) + 1e-8)
    print("Estimated global up (world):", global_up)

    # base 的 camera-to-world 旋轉
    R_wc_base = w2c_base[:3, :3].astype(np.float64)  # world -> camera
    R_c2w_base = R_wc_base.T                         # camera -> world

    # base 的「前方」方向（camera z 軸在 world 中）
    forward0 = R_c2w_base[:, 2]

    # 把 forward0 投影到 global_up 的水平面，去掉 pitch，使其「水平」
    forward0_proj = forward0 - np.dot(forward0, global_up) * global_up
    forward0_proj_norm = forward0_proj / (np.linalg.norm(forward0_proj) + 1e-8)

    # 透過 global_up 與 forward0_proj_norm 建立「扶正後」的 base 旋轉
    right0 = np.cross(global_up, forward0_proj_norm)
    right0 = right0 / (np.linalg.norm(right0) + 1e-8)
    up0 = np.cross(forward0_proj_norm, right0)

    R_c2w_base_level = np.stack([right0, up0, forward0_proj_norm], axis=1)

    # num_frames
    num_frames = max(2, int(args.num_frames))
    print(f"Generating {num_frames} frames for 360° horizontal rotation around scene center...")

    if args.out_dir is None:
        out_dir = model_path / "test_pack"
    else:
        out_dir = Path(args.out_dir).resolve()
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 根據 out_dir 路徑決定影片名稱
    video_name = "video.mp4"
    parts = set(out_dir.parts)
    if "baseline" in parts:
        video_name = "3dgs.mp4"
    elif "ours" in parts:
        video_name = "ours.mp4"

    print(f"PNG will be saved to : {png_dir}")
    print(f"MP4 will be saved to : {out_dir / video_name}")


    frames_for_video = []

    # 逐 frame 繞 global_up 做 360° 水平旋轉
    for idx in range(num_frames):
        theta = 2.0 * np.pi * idx / num_frames   # 0 ~ 2π

        # 在「世界座標」繞 global_up 旋轉
        R_yaw_world = axis_angle_rotation(global_up, theta)
        R_c2w_new = R_yaw_world @ R_c2w_base_level

        # world -> camera
        R_new = R_c2w_new.T.astype(np.float32)
        # t_new 讓 camera center 固定在 C
        t_new = (-R_new @ C.astype(np.float32))

        # 組成新的 w2c
        w2c_new = np.eye(4, dtype=np.float32)
        w2c_new[:3, :3] = R_new
        w2c_new[:3, 3]  = t_new

        # 渲染
        rgb, alpha_map, depth = er.render(w2c_new, K, H, W)
        rgb = rgb.clamp(0, 1).float()

        # 🔁 把畫面上下翻轉（修正 upside-down）
        rgb = torch.flip(rgb, dims=[1])  # [3, H, W]，維度 1 是垂直方向

        # 存 PNG
        frame_name = f"frame_{idx:04d}.png"
        frame_path = png_dir / frame_name
        torchvision.utils.save_image(rgb.cpu(), str(frame_path))

        # 收集給 mp4
        if HAS_IMAGEIO:
            frame = (rgb.cpu().numpy().transpose(1, 2, 0) * 255.0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames_for_video.append(frame)

        if idx % 20 == 0 or idx == num_frames - 1:
            print(f"Rendered frame {idx+1}/{num_frames} -> {frame_path.name}")

    # 輸出 mp4
    if HAS_IMAGEIO and len(frames_for_video) > 0:
        video_path = out_dir / video_name
        imageio.mimsave(video_path, frames_for_video, fps=args.fps)
        print(f"[Video] Saved mp4 to: {video_path}")

    elif not HAS_IMAGEIO:
        print("[Video] 跳過 mp4 輸出，因為沒有安裝 imageio（pip install imageio[ffmpeg]）")
    else:
        print("[Video] 沒有任何 frame，被跳過。")

    print(f"Done! Saved {num_frames} PNG images to: {png_dir}")

if __name__ == "__main__":
    main()
