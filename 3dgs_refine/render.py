#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer2 import render2
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer2 import GaussianModel_tam
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import numpy as np
from pathlib import Path
from pathlib import Path

def _build_K_W2C_from_view(view):
    # H, W
    H = int(getattr(view, "image_height", getattr(view, "height", 0)))
    W = int(getattr(view, "image_width",  getattr(view, "width",  0)))
    if H <= 0 or W <= 0:
        raise RuntimeError("Cannot get image size from view")

    
    #breakpoint()
    
    # fx, fy 優先用 focal，如果沒有就用 FovX/FovY 反推
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

    # W2C：3DGS 的 view 通常有 world_view_transform (4x4 world->camera)
    if hasattr(view, "world_view_transform"):
        w2c = view.world_view_transform
        if isinstance(w2c, torch.Tensor):
            w2c = w2c.detach().cpu().numpy()
        else:
            w2c = np.array(w2c, dtype=np.float32)
    else:
        raise RuntimeError("view missing world_view_transform")

    w2c = w2c.astype(np.float32)
    return w2c, K, H, W


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, er=None, mask_thresh=0.10):
    base = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(base, "renders")
    gts_path    = os.path.join(base, "gt")
    mask_path   = os.path.join(base, "mask") if er is not None else None

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if er is not None:
        makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        out_img = out["render"]
        gt      = view.original_image[0:3]

        if train_test_exp:
            out_img = out_img[..., out_img.shape[-1] // 2:]
            gt      = gt[...,      gt.shape[-1]      // 2:]

        if hasattr(view, "image_name") and isinstance(view.image_name, str) and view.image_name:
            stem = Path(view.image_name).stem
        else:
            stem = f"{idx:05d}"

        torchvision.utils.save_image(out_img, os.path.join(render_path, f"{stem}.png"))
        torchvision.utils.save_image(gt,      os.path.join(gts_path,    f"{stem}.png"))


            # 如果你也想存二值版，解除下面註解
            # bin_mask = (alpha_e >= mask_thresh).float()
            # torchvision.utils.save_image(bin_mask, os.path.join(mask_path, f"{stem}_bin.png"))




def render_set2(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, er=None, mask_thresh=0.10):
    base = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(base, "renders")
    gts_path    = os.path.join(base, "gt")
    mask_path   = os.path.join(base, "mask") if er is not None else None

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if er is not None:
        makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        #out = render2(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
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

        torchvision.utils.save_image(out_img, os.path.join(render_path, f"{stem}.png"))
        torchvision.utils.save_image(gt,      os.path.join(gts_path,    f"{stem}.png"))


            # 如果你也想存二值版，解除下面註解
            # bin_mask = (alpha_e >= mask_thresh).float()
            # torchvision.utils.save_image(bin_mask, os.path.join(mask_path, f"{stem}_bin.png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, skip_coarse: bool, separate_sh: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       er=None, mask_thresh=args.mask_thresh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       er=None, mask_thresh=args.mask_thresh)

        er = None
        
        #breakpoint()    
        
        #gaussians_2 = GaussianModel(args)
        
        if not skip_coarse:
            render_set2(dataset.model_path, "coarse", scene.loaded_iter, scene.getCoarseCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       er=er, mask_thresh=args.mask_thresh)
            
        
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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    #render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_coarse, SPARSE_ADAM_AVAILABLE)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args),
    #         args.skip_train, args.skip_test, args.skip_coarse, SPARSE_ADAM_AVAILABLE, args)
    render_sets(model.extract(args), args.total_iteration, pipeline.extract(args),
            args.skip_train, args.skip_test, args.skip_coarse, SPARSE_ADAM_AVAILABLE, args)

