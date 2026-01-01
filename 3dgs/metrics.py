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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir: Path, gt_dir: Path):
    """Read & align render/gt pairs. We sort by filename to ensure pairing."""
    renders, gts, image_names = [], [], []
    # 只收 png/jpg 並排序
    valid_ext = {".png", ".jpg", ".jpeg"}
    fnames = [f for f in os.listdir(renders_dir) if Path(f).suffix.lower() in valid_ext]
    fnames.sort()
    for fname in fnames:
        rpath = renders_dir / fname
        gpath = gt_dir / fname
        if not gpath.exists():
            # 若對應 GT 不存在，跳過
            continue
        render = Image.open(rpath)
        gt = Image.open(gpath)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    # 小包裝，避免重複打字
    def lpips_vgg(a, b):  return lpips(a, b, net_type='vgg')
    def lpips_alex(a, b): return lpips(a, b, net_type='alex')

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            if not test_dir.exists():
                print(f"  ⚠️ test dir not found: {test_dir}")
                continue

            for method in os.listdir(test_dir):
                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue
                print("Method:", method)

                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                if not (gt_dir.exists() and renders_dir.exists()):
                    print(f"  ⚠️ missing gt/renders: {gt_dir}, {renders_dir}")
                    continue

                renders, gts, image_names = readImages(renders_dir, gt_dir)
                if len(renders) == 0:
                    print("  ⚠️ no aligned image pairs. skip.")
                    continue

                ssims, psnrs = [], []
                lpips_vggs, lpips_alexs = [], []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    R = renders[idx]
                    G = gts[idx]
                    ssims.append(ssim(R, G))                 # scalar tensor
                    psnrs.append(psnr(R, G))                 # scalar tensor/float
                    lpips_vggs.append(lpips_vgg(R, G))       # scalar tensor
                    lpips_alexs.append(lpips_alex(R, G))     # scalar tensor

                # 轉成 tensor 再平均
                ssims_t       = torch.tensor([float(x) for x in ssims], device='cuda')
                psnrs_t       = torch.tensor([float(x) for x in psnrs], device='cuda')
                lpips_vgg_t   = torch.tensor([float(x) for x in lpips_vggs], device='cuda')
                lpips_alex_t  = torch.tensor([float(x) for x in lpips_alexs], device='cuda')

                print("  SSIM        : {:>12.7f}".format(ssims_t.mean().item()))
                print("  PSNR        : {:>12.7f}".format(psnrs_t.mean().item()))
                print("  LPIPS (VGG) : {:>12.7f}".format(lpips_vgg_t.mean().item()))
                print("  LPIPS (Alex): {:>12.7f}".format(lpips_alex_t.mean().item()))
                print("")

                # 寫 scene-level 平均
                full_dict[scene_dir][method] = {
                    "SSIM":        ssims_t.mean().item(),
                    "PSNR":        psnrs_t.mean().item(),
                    "LPIPS_VGG":   lpips_vgg_t.mean().item(),
                    "LPIPS_ALEX":  lpips_alex_t.mean().item(),
                }

                # 寫 per-view
                per_view_dict[scene_dir][method] = {
                    "SSIM":       {name: float(val) for name, val in zip(image_names, ssims_t.tolist())},
                    "PSNR":       {name: float(val) for name, val in zip(image_names, psnrs_t.tolist())},
                    "LPIPS_VGG":  {name: float(val) for name, val in zip(image_names, lpips_vgg_t.tolist())},
                    "LPIPS_ALEX": {name: float(val) for name, val in zip(image_names, lpips_alex_t.tolist())},
                }

            # 各 scene 輸出 JSON
            out_avg = Path(scene_dir) / "results.json"
            out_per = Path(scene_dir) / "per_view.json"
            with open(out_avg, 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=2)
            with open(out_per, 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=2)
            print(f"  ✔ wrote {out_avg}")
            print(f"  ✔ wrote {out_per}")

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("  Error:", repr(e))

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
