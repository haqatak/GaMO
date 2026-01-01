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


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (qw, qx, qy, qz)"""
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[1, 0] + R[0, 1]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[2, 0] + R[0, 2]) / s
            q[2] = (R[2, 1] + R[1, 2]) / s
            q[3] = 0.25 * s
    return q


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
    回傳:
      (w2c, K, H, W, name, qvec, tvec)
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

        q = meta["qvec"]
        t = meta["tvec"]
        R = qvec2rotmat(q)  # world->camera
        t_vec = t.reshape(3, 1)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3]  = t_vec[:, 0].astype(np.float32)

        items.append((w2c, K, H, W, name, q, t))
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
# Quaternion SLERP
# ----------------------------
def slerp_quat(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions.
    q0, q1: (4,) array [qw, qx, qy, qz]
    t: interpolation factor [0, 1]
    """
    dot = np.dot(q0, q1)
    
    # If dot < 0, negate q1 to take shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Clamp dot to avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # Compute the angle between quaternions
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_poses(q0, t0, q1, t1, alpha):
    """
    在兩個 pose 之間插值
    q0, q1: quaternion (4,) [qw, qx, qy, qz]
    t0, t1: translation (3,)
    alpha: interpolation factor [0, 1]
    
    返回: (q_interp, t_interp)
    """
    q_interp = slerp_quat(q0, q1, alpha)
    t_interp = (1 - alpha) * t0 + alpha * t1
    return q_interp, t_interp


# ----------------------------
# Generate interpolated trajectory
# ----------------------------
def smooth_velocity_curve(t, slow_ratio=0.3):
    """
    生成平滑的速度曲線，在 0 和 1 附近放慢，中間保持較快速度
    
    Args:
        t: 輸入參數 [0, 1]
        slow_ratio: 在兩端放慢的區域比例（0-0.5）
    
    Returns:
        平滑映射後的值 [0, 1]
    """
    if slow_ratio <= 0:
        # 完全線性，無速度變化
        return t
    
    if t <= slow_ratio:
        # 前段：使用 smoothstep (3t^2 - 2t^3) 更平滑
        normalized = t / slow_ratio
        smoothed = normalized * normalized * (3 - 2 * normalized)
        return 0.5 * slow_ratio * smoothed
    elif t >= (1 - slow_ratio):
        # 後段：使用 smoothstep
        normalized = (t - (1 - slow_ratio)) / slow_ratio
        smoothed = normalized * normalized * (3 - 2 * normalized)
        return 1 - 0.5 * slow_ratio * (1 - smoothed)
    else:
        # 中段：線性移動（較快）
        mid_start = 0.5 * slow_ratio
        mid_end = 1 - 0.5 * slow_ratio
        mid_length = mid_end - mid_start
        
        progress = (t - slow_ratio) / (1 - 2 * slow_ratio)
        return mid_start + progress * mid_length


def generate_interpolated_trajectory(keyframes, min_total_frames=150, 
                                     frames_per_segment=None, slow_ratio=0.15):
    """
    在關鍵 frame 之間生成平滑內插軌跡（改進版：更自然的速度變化）
    
    Args:
        keyframes: list of (q, t, name) - 關鍵幀的 pose
        min_total_frames: 最小總幀數
        frames_per_segment: 每段的幀數（None 則自動計算）
        slow_ratio: 在關鍵幀附近放慢的區域比例（0-0.5，預設 0.15；設為 0 則完全勻速）
    
    Returns:
        list of (q, t, frame_type, keyframe_idx, frame_name) 
    """
    if len(keyframes) < 2:
        raise ValueError("至少需要 2 個關鍵幀")
    
    n_segments = len(keyframes) - 1
    
    # 自動計算每段需要多少幀
    if frames_per_segment is None:
        frames_per_segment = max(30, min_total_frames // n_segments)
    
    trajectory = []
    use_constant_speed = (slow_ratio < 0.01)  # 幾乎為 0 時使用完全勻速
    
    for i in range(n_segments):
        q_curr, t_curr, name_curr = keyframes[i]
        q_next, t_next, name_next = keyframes[i + 1]
        
        # 生成這一段的所有幀
        for j in range(frames_per_segment):
            # 線性進度 [0, 1]
            t_linear = j / frames_per_segment
            
            # 應用平滑速度曲線（或使用勻速）
            if use_constant_speed:
                t_smooth = t_linear  # 完全勻速
                frame_type = 'linear'
            else:
                t_smooth = smooth_velocity_curve(t_linear, slow_ratio)
                # 判斷幀類型（用於 debug）
                if j == 0:
                    frame_type = 'key'
                elif t_linear < slow_ratio or t_linear > (1 - slow_ratio):
                    frame_type = 'slow'
                else:
                    frame_type = 'fast'
            
            # 插值 pose
            q_interp, t_interp = interpolate_poses(q_curr, t_curr, q_next, t_next, t_smooth)
            
            if j == 0:
                frame_name = name_curr
            else:
                frame_name = f"{name_curr}_to_{name_next}"
            
            trajectory.append((q_interp, t_interp, frame_type, i, frame_name))
    
    # 添加最後一個關鍵幀
    q_last, t_last, name_last = keyframes[-1]
    trajectory.append((q_last, t_last, 'key', n_segments, name_last))
    
    # 檢查總幀數，如果不夠就增加每段的幀數
    current_frames = len(trajectory)
    if current_frames < min_total_frames:
        needed_per_segment = (min_total_frames - 1) // n_segments + 1
        print(f"當前幀數 {current_frames} < {min_total_frames}，增加每段幀數到 {needed_per_segment}")
        return generate_interpolated_trajectory(keyframes, min_total_frames, 
                                               needed_per_segment, slow_ratio)
    
    return trajectory


# ----------------------------
# Main
# ----------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument("--source_path", required=True,
                    help="資料根目錄（底下有 sparse/ 和 images_test_select/）")

    # 預設讀 test
    ap.add_argument("--images_subdir", default="test",
                    help="讀 images.txt 的子資料夾（預設 sparse/test）")
    ap.add_argument("--cameras_subdir", default="test",
                    help="讀 cameras 的子資料夾（預設 sparse/test）")
    ap.add_argument("--cameras_filename", default="cameras.txt",
                    help="cameras 檔名（預設 cameras.txt）")
    
    # 篩選圖片的資料夾
    ap.add_argument("--select_dir", default="images_test_select",
                    help="篩選後的圖片資料夾名稱（預設 images_test_select）")
    
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
                    help="render RGB 輸出資料夾（未指定則丟到 model_path/test_pack_interpolated）")

    # mp4 fps
    ap.add_argument("--fps", type=int, default=30,
                    help="輸出 mp4 的 FPS（預設 30）")

    # 內插參數
    ap.add_argument("--min_frames", type=int, default=200,
                    help="最小總幀數（預設 200，增加可讓速度變化更細膩）")
    ap.add_argument("--frames_per_segment", type=int, default=None,
                    help="每段的幀數（None 則根據 min_frames 自動計算）")
    ap.add_argument("--slow_ratio", type=float, default=0.15,
                    help="在關鍵幀附近放慢的區域比例 [0-0.5]（預設 0.15，即前後各 15%% 會放慢；設為 0 則完全勻速）")

    # Floater 過濾參數
    ap.add_argument("--filter_floaters", action="store_true",
                    help="啟用 floater 過濾（移除飄浮的 Gaussians）")
    ap.add_argument("--opacity_threshold", type=float, default=0.1,
                    help="透明度閾值，低於此值的 Gaussians 會被過濾（預設 0.1）")
    ap.add_argument("--scale_threshold", type=float, default=None,
                    help="Scale 閾值，大於此值的 Gaussians 會被過濾（預設 None 不過濾）")
    ap.add_argument("--depth_near", type=float, default=None,
                    help="最近深度，小於此值的點會被過濾（預設 None）")
    ap.add_argument("--depth_far", type=float, default=None,
                    help="最遠深度，大於此值的點會被過濾（預設 None）")

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

    # Floater 過濾
    if args.filter_floaters:
        print("\n" + "="*50)
        print("[Floater Filtering] Enabled")
        print("="*50)
        
        try:
            # 方法 1：使用簡單的過濾（推薦）
            original_count = len(er.gaussians.get_xyz)
            
            # 建立過濾 mask
            mask = torch.ones(original_count, dtype=torch.bool, device=er.gaussians.get_xyz.device)
            
            # 透明度過濾
            if args.opacity_threshold > 0:
                opacity = er.gaussians.get_opacity.squeeze()
                opacity_mask = opacity >= args.opacity_threshold
                filtered = (~opacity_mask).sum().item()
                mask = mask & opacity_mask
                print(f"  ✓ Opacity filter: removed {filtered} low-opacity Gaussians (threshold={args.opacity_threshold})")
            
            # Scale 過濾
            if args.scale_threshold is not None:
                scaling = er.gaussians.get_scaling
                max_scale = scaling.max(dim=1)[0]
                scale_mask = max_scale <= args.scale_threshold
                filtered = (~scale_mask).sum().item()
                mask = mask & scale_mask
                print(f"  ✓ Scale filter: removed {filtered} large Gaussians (threshold={args.scale_threshold})")
            
            # 深度過濾
            if args.depth_near is not None or args.depth_far is not None:
                xyz = er.gaussians.get_xyz
                scene_center = xyz.mean(dim=0)
                distances = torch.norm(xyz - scene_center, dim=1)
                
                if args.depth_near is not None:
                    near_mask = distances >= args.depth_near
                    filtered = (~near_mask).sum().item()
                    mask = mask & near_mask
                    print(f"  ✓ Near depth filter: removed {filtered} Gaussians (threshold={args.depth_near})")
                
                if args.depth_far is not None:
                    far_mask = distances <= args.depth_far
                    filtered = (~far_mask).sum().item()
                    mask = mask & far_mask
                    print(f"  ✓ Far depth filter: removed {filtered} Gaussians (threshold={args.depth_far})")
            
            # 統計
            total_filtered = (~mask).sum().item()
            remaining = mask.sum().item()
            
            if total_filtered > 0:
                print(f"\n  📊 Summary:")
                print(f"     Original:  {original_count:,} Gaussians")
                print(f"     Filtered:  {total_filtered:,} ({total_filtered/original_count*100:.1f}%)")
                print(f"     Remaining: {remaining:,} Gaussians")
                
                # 應用過濾（修改 Gaussians）
                # 注意：這可能需要根據 EasyRenderer 的實際實作調整
                try:
                    er.gaussians._xyz = er.gaussians._xyz[mask]
                    er.gaussians._features_dc = er.gaussians._features_dc[mask]
                    er.gaussians._features_rest = er.gaussians._features_rest[mask]
                    er.gaussians._scaling = er.gaussians._scaling[mask]
                    er.gaussians._rotation = er.gaussians._rotation[mask]
                    er.gaussians._opacity = er.gaussians._opacity[mask]
                    print(f"  ✅ Filtering applied successfully!\n")
                except AttributeError as e:
                    print(f"  ⚠️  Warning: Cannot directly modify Gaussians")
                    print(f"     Error: {e}")
                    print(f"     The renderer may not support in-place filtering.")
                    print(f"     Rendering will proceed with unfiltered Gaussians.\n")
            else:
                print(f"  ℹ️  No Gaussians were filtered with current thresholds\n")
                
        except Exception as e:
            print(f"  ❌ Error during filtering: {e}")
            print(f"     Rendering will proceed with unfiltered Gaussians.\n")
    else:
        print("\n[Floater Filtering] Disabled")
        print("  💡 Use --filter_floaters to enable floater removal")
        print("  💡 Typical usage: --filter_floaters --opacity_threshold 0.1 --scale_threshold 0.5\n")


    # 載入所有 poses（images.txt + cameras.txt）
    all_items = load_poses(
        source_path=str(src),
        images_subdir=args.images_subdir,
        cameras_subdir=args.cameras_subdir,
        cameras_filename=args.cameras_filename,
    )
    print(f"Loaded {len(all_items)} poses from {src}/sparse/{args.images_subdir}/images.txt")

    # 檢查 images_test_select 資料夾，找出實際存在的圖片
    select_dir = src / args.select_dir
    if not select_dir.exists():
        raise FileNotFoundError(f"篩選圖片資料夾不存在: {select_dir}")
    
    # 獲取所有篩選後的圖片檔名
    selected_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        selected_images.update([p.name for p in select_dir.glob(ext)])
    
    print(f"Found {len(selected_images)} selected images in {select_dir}")

    # 篩選出存在於 images_test_select 中的 pose
    filtered_items = []
    for item in all_items:
        w2c, K, H, W, name, q, t = item
        if name in selected_images:
            filtered_items.append(item)
    
    if len(filtered_items) < 2:
        raise RuntimeError(f"篩選後至少需要 2 個 pose，但只找到 {len(filtered_items)} 個")
    
    # 按檔名排序
    filtered_items = sorted(filtered_items, key=lambda x: x[4])
    print(f"Filtered to {len(filtered_items)} poses that exist in {args.select_dir}/")
    print("Selected keyframes:")
    for idx, (_, _, _, _, name, _, _) in enumerate(filtered_items):
        print(f"  {idx}: {name}")

    # intrinsics / resolution 用第一個 view，並把 FOV 拉廣
    w2c0, K_base, H_base, W_base, name0, q0, t0 = filtered_items[0]
    K = K_base.copy().astype(np.float32)
    K[0, 0] *= 0.8  # fx
    K[1, 1] *= 0.8  # fy
    H, W = H_base, W_base
    print(f"\nBase view (for intrinsics): {name0}")
    print(f"Scaled intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")

    # 準備關鍵幀資料
    keyframes = [(item[5], item[6], item[4]) for item in filtered_items]  # (q, t, name)
    
    # 生成內插軌跡
    print(f"\nGenerating interpolated trajectory...")
    print(f"  slow_ratio={args.slow_ratio} (前後各 {args.slow_ratio*100:.0f}% 區域會放慢)")
    if args.frames_per_segment:
        print(f"  frames_per_segment={args.frames_per_segment}")
    trajectory = generate_interpolated_trajectory(
        keyframes, 
        min_total_frames=args.min_frames,
        frames_per_segment=args.frames_per_segment,
        slow_ratio=args.slow_ratio
    )
    
    total_frames = len(trajectory)
    print(f"Generated {total_frames} frames (target: >={args.min_frames})")

    # 輸出資料夾
    if args.out_dir is None:
        out_dir = model_path / "test_pack_interpolated"
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

    print(f"\nPNG will be saved to : {png_dir}")
    print(f"MP4 will be saved to : {out_dir / video_name}")
    print(f"Video duration: ~{total_frames/args.fps:.1f} seconds @ {args.fps} fps\n")

    frames_for_video = []

    # 渲染所有幀
    for idx, (q, t, frame_type, key_idx, frame_name) in enumerate(trajectory):
        # 從 quaternion + translation 構建 w2c
        R = qvec2rotmat(q)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3] = t.astype(np.float32)

        # 渲染
        rgb, alpha_map, depth = er.render(w2c, K, H, W)
        rgb = rgb.clamp(0, 1).float()

        # 存 PNG
        frame_filename = f"frame_{idx:04d}.png"
        frame_path = png_dir / frame_filename
        torchvision.utils.save_image(rgb.cpu(), str(frame_path))

        # 收集給 mp4
        if HAS_IMAGEIO:
            frame = (rgb.cpu().numpy().transpose(1, 2, 0) * 255.0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames_for_video.append(frame)

        # 進度顯示
        if idx % 20 == 0 or idx == total_frames - 1:
            type_str = {'key': 'KEY', 'slow': 'SLOW', 'fast': 'FAST', 'linear': 'LINEAR'}[frame_type]
            print(f"Rendered {idx+1}/{total_frames} [{type_str}] segment={key_idx} -> {frame_filename}")

    # 輸出 mp4
    if HAS_IMAGEIO and len(frames_for_video) > 0:
        video_path = out_dir / video_name
        imageio.mimsave(video_path, frames_for_video, fps=args.fps)
        duration = len(frames_for_video) / args.fps
        print(f"\n[Video] Saved mp4 to: {video_path}")
        print(f"        Duration: {duration:.2f} seconds ({len(frames_for_video)} frames @ {args.fps} fps)")
    elif not HAS_IMAGEIO:
        print("\n[Video] 跳過 mp4 輸出，因為沒有安裝 imageio（pip install imageio[ffmpeg]）")
    else:
        print("\n[Video] 沒有任何 frame，被跳過。")

    print(f"\nDone! Saved {total_frames} PNG images to: {png_dir}")
    print("\nTrajectory summary:")
    print(f"  Total frames: {total_frames}")
    print(f"  Keyframes: {len(keyframes)}")
    print(f"  Segments: {len(keyframes) - 1}")
    print(f"  Avg frames per segment: {total_frames / (len(keyframes) - 1):.1f}")
    print(f"  Slow ratio: {args.slow_ratio} (前後各 {args.slow_ratio*100:.0f}% 會放慢)")

if __name__ == "__main__":
    main()