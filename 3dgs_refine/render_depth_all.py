#!/usr/bin/env python3
"""
Universal Depth Renderer for 3D Gaussian Splatting models
支援不同版本/方法訓練的模型,只需要:
1. checkpoint (.ply 或 point_cloud.ply)
2. camera poses (COLMAP format 或 transforms.json)
3. 場景參數 (可選)
"""

import os
import sys
import json
import torch
import numpy as np
import torchvision
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
class CompatPipeline:
    """
    盡量相容各家 3DGS render() 會讀的 pipeline 旗標。
    找不到的屬性一律回傳 None，避免 AttributeError。
    """
    def __init__(self, **kwargs):
        # 常見旗標的保守預設
        defaults = dict(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            use_confidence=False,      # 你的錯誤訊息指名要這個
            antialiasing=False,
            opacity_threshold=0.0,
            override_color=None,
            eval=False,
            render_mode=None,
            # 某些分支會讀到的
            enable_sh_degree_variation=False,
            temporal_id=None,
            supersampling=1,           # 偶爾會讀這個；1=不超採樣
        )
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        # 任何未知旗標一律回 None，避免 AttributeError
        return None

# ==================== 通用相機類 ====================
class Camera:
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height, image_name=""):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.image_name = image_name
        
        # 計算投影矩陣
        self.world_view_transform = self._getWorld2View2(R, T).transpose(0, 1)
        self.projection_matrix = self._getProjectionMatrix(FoVx, FoVy, 0.01, 100.0).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return torch.from_numpy(Rt).float()
    
    def _getProjectionMatrix(self, fovX, fovY, znear, zfar):
        tanHalfFovY = np.tan((fovY / 2))
        tanHalfFovX = np.tan((fovX / 2))
        
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        
        P = torch.zeros(4, 4)
        z_sign = 1.0
        
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

# ==================== 讀取 Cameras ====================
def load_cameras_from_colmap(sparse_path: str, images_folder: str = "images"):
    """從 COLMAP sparse/0 讀取相機 (支援 .bin 和 .txt 格式)"""
    try:
        from scene.colmap_loader import (
            read_extrinsics_binary, read_intrinsics_binary,
            read_extrinsics_text, read_intrinsics_text
        )
    except:
        print("Warning: Cannot import COLMAP loader")
        return []
    
    # 優先嘗試 .bin 格式,如果不存在則使用 .txt
    cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
    cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
    
    if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
        print("Loading from binary format (.bin)")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    else:
        # 嘗試 .txt 格式
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        
        if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
            print("Loading from text format (.txt)")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        else:
            print(f"Error: Cannot find cameras/images files in {sparse_path}")
            print(f"  Tried: images.bin, images.txt, cameras.bin, cameras.txt")
            return []
    
    cameras = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        height = intr.height
        width = intr.width
        
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = 2 * np.arctan(height / (2 * focal_length_x))
            FovX = 2 * np.arctan(width / (2 * focal_length_x))
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = 2 * np.arctan(height / (2 * focal_length_y))
            FovX = 2 * np.arctan(width / (2 * focal_length_x))
        else:
            focal_length_x = intr.params[0]
            FovY = 2 * np.arctan(height / (2 * focal_length_x))
            FovX = 2 * np.arctan(width / (2 * focal_length_x))
        
        cam = Camera(R, T, FovX, FovY, width, height, extr.name)
        cameras.append(cam)
    
    return cameras

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

# ==================== 讀取 Gaussians ====================
# ==================== 讀取 Gaussians ====================
def load_gaussians_from_ply(ply_path: str, device="cuda"):
    """
    用 Graphdeco 的 GaussianModel 讀取 .ply，
    回傳真正的 GaussianModel 物件（render() 才吃得下）
    """
    # 以球諧階數=3 初始化（和 3DGS 預設一致）
    try:
        from gaussian_renderer import GaussianModel
    except Exception as e:
        raise RuntimeError(f"無法匯入 GaussianModel：{e}")

    gm = GaussianModel(sh_degree=3)
    gm.load_ply(ply_path)
    # 注意：GaussianModel 本身會把資料放在正確的 device（通常是 CUDA）
    return gm


# ==================== 簡化的 Depth Renderer ====================
def render_depth_simple(camera: Camera, gaussians, background: torch.Tensor):
    try:
        from gaussian_renderer import render

        pipe = CompatPipeline()  # ← 用相容管線

        # background 要是 [3] 的 CUDA float32
        bg = background
        if not isinstance(bg, torch.Tensor):
            bg = torch.tensor(bg, dtype=torch.float32)
        if bg.ndim != 1 or bg.numel() != 3:
            bg = bg.flatten()[:3]
        bg = bg.to(dtype=torch.float32, device="cuda")

        out = render(camera, gaussians, pipe, bg)

        # 大多數分支回 dict-like
        depth = out.get("depth", None) if isinstance(out, dict) else None
        return depth

    except Exception as e:
        print(f"Render failed: {e}")
        return None



# ==================== Depth 可視化 ====================
def depth_to_heatmap(depth: torch.Tensor, cmap="turbo") -> torch.Tensor:
    """轉換 depth 為彩色熱力圖"""
    d = depth.detach().cpu().float()
    
    # 過濾無效值
    valid = torch.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return torch.ones(3, d.shape[0], d.shape[1]) * 0.5
    
    # 正規化
    dmin, dmax = d[valid].min(), d[valid].max()
    dn = torch.clamp((d - dmin) / (dmax - dmin + 1e-8), 0, 1)
    dn = 1.0 - dn  # 近亮遠暗
    
    # 應用 colormap
    dn_np = dn.numpy()
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(dn_np)[..., :3]).astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)

# ==================== 主函數 ====================
def main():
    parser = ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to .ply checkpoint")
    parser.add_argument("--cameras", type=str, required=True, help="Path to COLMAP sparse/0 or transforms.json")
    parser.add_argument("--output", type=str, default="./depth_output", help="Output directory")
    parser.add_argument("--white_bg", action="store_true", help="Use white background")
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output, exist_ok=True)
    
    # 讀取 Gaussians
    print(f"Loading Gaussians from {args.ply}")
    gaussians = load_gaussians_from_ply(args.ply)  # ← 回傳的是 GaussianModel，不是 dict
    
    # 讀取 Cameras
    print(f"Loading cameras from {args.cameras}")
    cameras = load_cameras_from_colmap(args.cameras)
    print(f"Found {len(cameras)} cameras")
    
    # 設定背景
    bg_color = [1, 1, 1] if args.white_bg else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Render
    for idx, cam in enumerate(tqdm(cameras, desc="Rendering depth")):
        depth = render_depth_simple(cam, gaussians, background)
        
        if depth is not None:
            heat = depth_to_heatmap(depth)
            stem = Path(cam.image_name).stem if cam.image_name else f"{idx:05d}"
            torchvision.utils.save_image(heat, os.path.join(args.output, f"{stem}_depth.png"))
    
    print(f"Done! Depth maps saved to {args.output}")

if __name__ == "__main__":
    main()