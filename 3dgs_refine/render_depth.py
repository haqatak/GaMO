# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from os import makedirs
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from gaussian_renderer2 import GaussianModel_tam, render2  # noqa: F401 (keep if you need render2 path)
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# ================================
# 可視化設定（兩邊對齊用）
# ================================
VIS_CLIP_LOW  = 5.0
VIS_CLIP_HIGH = 95.0
VIS_LOG       = True
VIS_INVERT    = True        # 近紅遠藍（invert 後近亮）
VIS_GAMMA     = 0.9
VIS_CMAP      = "turbo"     # 或 "magma"/"viridis" 皆可，但兩邊要一致

def vis_depth_like_left(
    depth_hw,
    clip_low: float = VIS_CLIP_LOW,
    clip_high: float = VIS_CLIP_HIGH,
    log_vis: bool = VIS_LOG,
    invert: bool = VIS_INVERT,
    gamma: float = VIS_GAMMA,
    cmap_name: str = VIS_CMAP,
    valid_mask: Optional[np.ndarray] = None,
    fixed_clip: Optional[tuple] = None,   # (lo, hi) 或 None
) -> np.ndarray:
    """
    輸入: depth_hw 可為 torch.Tensor(H,W) 或 np.ndarray(H,W)
    回傳: BGR uint8 (H,W,3)，可直接 cv2.imwrite
    """

    # 1) 先把 depth 轉成 numpy float32 —— 一定要在任何用到 d 之前
    if isinstance(depth_hw, torch.Tensor):
        d = depth_hw.detach().cpu().float().numpy()
    else:
        d = np.asarray(depth_hw, dtype=np.float32)

    # 2) 構建有效遮罩（需要 d）
    if valid_mask is None:
        valid = np.isfinite(d) & (d > 0) & (d < 1e6)
    else:
        vm = np.asarray(valid_mask, dtype=bool)
        # 若遮罩尺寸不符（例如先裁切過 depth），做個保護
        if vm.shape != d.shape:
            # 嘗試中心對齊裁切到同尺寸；不行就退回只看 d 的有限性
            try:
                h, w = d.shape
                vh, vw = vm.shape
                top = max((vh - h) // 2, 0)
                left = max((vw - w) // 2, 0)
                vm = vm[top:top + h, left:left + w]
                if vm.shape != d.shape:
                    vm = None
            except Exception:
                vm = None
        valid = (np.isfinite(d) if vm is None else vm & np.isfinite(d))

    # 3) 若沒有有效像素，回傳保底灰圖
    if valid.size == 0 or valid.sum() == 0:
        return np.full((*d.shape, 3), 128, np.uint8)

    # 4) 取得 (lo, hi)：優先 fixed_clip，否則用百分位
    if fixed_clip is not None:
        lo, hi = float(fixed_clip[0]), float(fixed_clip[1])
    else:
        flat = d[valid]
        lo = np.percentile(flat, clip_low)
        hi = np.percentile(flat, clip_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(flat.min()), float(flat.max())
            if hi - lo < 1e-6:
                m = 0.5 * (hi + lo)
                lo, hi = m - 0.01, m + 0.01

    # 5) 正規化 + 視覺化流程
    d = np.clip(d, lo, hi)
    dn = (d - lo) / (hi - lo + 1e-12)

    if log_vis:
        dn = np.log1p(dn * 9.0) / np.log(10.0)

    if invert:
        dn = 1.0 - dn

    dn = np.clip(dn, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        dn = dn ** gamma

    dn[~valid] = 0.0

    rgb = plt.get_cmap(cmap_name)(dn)[..., :3]  # RGB in [0,1]
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return rgb_u8[..., ::-1]  # BGR


def _build_K_W2C_from_view(view):
    # H, W
    H = int(getattr(view, "image_height", getattr(view, "height", 0)))
    W = int(getattr(view, "image_width",  getattr(view, "width",  0)))
    if H <= 0 or W <= 0:
        raise RuntimeError("Cannot get image size from view")

    # fx, fy 優先用 focal，否則由 FovX/FovY 反推
    if hasattr(view, "focal_x") and hasattr(view, "focal_y"):
        fx = float(view.focal_x); fy = float(view.focal_y)
    else:
        FovX = float(view.FoVx); FovY = float(view.FoVy)
        fx = 0.5 * W / np.tan(0.5 * FovX)
        fy = 0.5 * H / np.tan(0.5 * FovY)

    cx = 0.5 * W
    cy = 0.5 * H
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    # world -> camera
    if hasattr(view, "world_view_transform"):
        w2c = view.world_view_transform
        if isinstance(w2c, torch.Tensor):
            w2c = w2c.detach().cpu().numpy()
        else:
            w2c = np.array(w2c, dtype=np.float32)
    else:
        raise RuntimeError("view missing world_view_transform")

    return w2c.astype(np.float32), K, H, W


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, er=None, mask_thresh=0.10):
    """
    主渲染（細緻）路徑：輸出彩色深度到 {model_path}/{name}/ours_{iter}/depth/*.png
    並與你另一支腳本保持一致：同參數、同遮罩、同裁切、同存法。
    """
    base = os.path.join(model_path, name, f"ours_{iteration}")
    depth_dir = os.path.join(base, "depth")
    gts_dir   = os.path.join(base, "gt")

    makedirs(depth_dir, exist_ok=True)
    makedirs(gts_dir,    exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        out = render(view, gaussians, pipeline, background,
                     use_trained_exp=train_test_exp, separate_sh=separate_sh)

        depth = out["depth"]
        if isinstance(depth, torch.Tensor) and depth.dim() == 3 and depth.shape[0] == 1:
            depth = depth[0]
        elif isinstance(depth, torch.Tensor) and depth.dim() != 2:
            raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

        # ===== 1) 先做裁切 =====
        if train_test_exp:
            depth = depth[:, depth.shape[1] // 2:]

        # ===== 2) 明確處理單位 =====
        if args.depth_unit == "inverse":
            depth = 1.0 / torch.clamp(depth, min=1e-8)

        # ===== 3) 同步產生 valid_mask（裁切後才算）=====
        valid_mask = None
        if "alpha" in out:
            alpha = out["alpha"]
            if isinstance(alpha, torch.Tensor) and alpha.dim() == 3 and alpha.shape[0] == 1:
                alpha = alpha[0]
            if train_test_exp:
                alpha = alpha[:, alpha.shape[1] // 2:]
            valid_mask = (alpha > args.alpha_thresh).detach().cpu().numpy()

        # ===== 4) 決定 clip 模式 =====
        fixed_clip = None
        if args.clip_mode == "global":
            assert (args.global_lo is not None) and (args.global_hi is not None), \
                "global clip 模式需要 --global_lo 與 --global_hi"
            fixed_clip = (float(args.global_lo), float(args.global_hi))

        # ===== 5) 視覺化（兩邊共用同一函數 + 同參數）=====
        depth_bgr = vis_depth_like_left(
            depth_hw=depth,
            clip_low=VIS_CLIP_LOW, clip_high=VIS_CLIP_HIGH,
            log_vis=VIS_LOG, invert=VIS_INVERT, gamma=VIS_GAMMA,
            cmap_name=VIS_CMAP,
            valid_mask=valid_mask,
            fixed_clip=fixed_clip,  # ← 如需全域一致 clip
        )
        
        # 檔名（可選：若另一支有加 _depth 後綴，這裡也加）
        if hasattr(view, "image_name") and isinstance(view.image_name, str) and view.image_name:
            stem = Path(view.image_name).stem
        else:
            stem = f"{idx:05d}"

        png_path = os.path.join(depth_dir, f"{stem}_depth.png")
        npy_path = os.path.join(depth_dir, f"{stem}_depth.npy")  # 也存 raw 供對照（可關）
        cv2.imwrite(png_path, depth_bgr)
        np.save(npy_path, depth.detach().cpu().numpy())

        # --- （可選）存對應 GT 影像，保持行為與原版一致 ---
        gt = view.original_image[0:3]
        if train_test_exp:
            gt = gt[..., gt.shape[-1] // 2:]
        torchvision.utils.save_image(gt, os.path.join(gts_dir, f"{stem}.png"))


def render_set2(model_path, name, iteration, views, gaussians, pipeline, background,
                train_test_exp, separate_sh, er=None, mask_thresh=0.10):
    """
    coarse 路徑，保留原 render2 出圖行為（非 depth）。
    """
    base = os.path.join(model_path, name, f"ours_{iteration}")
    render_dir = os.path.join(base, "renders")
    gts_dir    = os.path.join(base, "gt")
    mask_dir   = os.path.join(base, "mask") if er is not None else None

    makedirs(render_dir, exist_ok=True)
    makedirs(gts_dir,    exist_ok=True)
    if er is not None:
        makedirs(mask_dir, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        out = render2(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background
        )

        out_img = out["render"]
        gt      = view.original_image[0:3]

        if train_test_exp:
            out_img = out_img[..., out_img.shape[-1] // 2:]
            gt      = gt[...,      gt.shape[-1]      // 2:]

        if hasattr(view, "image_name") and isinstance(view.image_name, str) and view.image_name:
            stem = Path(view.image_name).stem
        else:
            stem = f"{idx:05d}"

        torchvision.utils.save_image(out_img, os.path.join(render_dir, f"{stem}.png"))
        torchvision.utils.save_image(gt,      os.path.join(gts_dir,    f"{stem}.png"))

        # 若未來要輸出 mask，可在此加入（目前保留原邏輯）。


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, skip_coarse: bool, separate_sh: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       er=None, mask_thresh=args.mask_thresh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       er=None, mask_thresh=args.mask_thresh)

        # coarse（非 depth）路徑
        if not skip_coarse:
            render_set2(dataset.model_path, "coarse", scene.loaded_iter, scene.getCoarseCameras(),
                        gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                        er=None, mask_thresh=args.mask_thresh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--total_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_coarse", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mask_with_easy", action="store_true",
                        help="在 coarse pass 另外用 EasyRenderer 產生 mask")
    parser.add_argument("--mask_thresh", type=float, default=0.10,
                        help="alpha 二值化閾值（輸出二值遮罩時用；soft mask 會輸出 1-alpha）")
    
    # 在 __main__ 的 parser 區塊加：
    parser.add_argument("--depth_unit", choices=["metric","inverse"], default="metric",
                        help="明確指定輸入 depth 單位；inverse 表示需要 1/depth 轉回 metric")
    parser.add_argument("--alpha_thresh", type=float, default=0.01,
                        help="alpha 轉 valid_mask 的閾值（兩邊一致）")
    parser.add_argument("--clip_mode", choices=["per_image","global"], default="per_image",
                        help="per_image: 用百分位 5/95；global: 用 --global_lo/--global_hi")
    parser.add_argument("--global_lo", type=float, default=None,
                        help="clip_mode=global 時使用的 lo（公尺等 metric 單位）")
    parser.add_argument("--global_hi", type=float, default=None,
                        help="clip_mode=global 時使用的 hi")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.total_iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.skip_coarse, SPARSE_ADAM_AVAILABLE, args)
