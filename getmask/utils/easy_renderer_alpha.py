import os, sys
sys.path.append("../")
import numpy as np
import torch
import torchvision
from pathlib import Path

from scene import Scene
from scene.cameras import PseudoCamera
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args_without_cmdlne


class EasyRenderer:
    def __init__(self, model_path, iteration=10000):
        """
        model_path: 3DGS / ours 的 model 資料夾
                    e.g. /project2/.../output/Duster/Replica_3/office_2/20251110-131820
        iteration:  要載入的 iteration 編號
                    - >0 : 用指定的 iteration_xxxx
                    - <=0: 自動找最新的 iteration_xxxx
        """
        self.model_path = Path(model_path)
        self.iteration = int(iteration)

        parser = ArgumentParser(description="Easy renderer")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        args = get_combined_args_without_cmdlne(parser, trained_model_path=model_path)

        model_param = model.extract(args)
        pipeline_param = pipeline.extract(args)

        # 把 iteration 塞回 args，讓 GaussianModel 有機會用到
        args.iteration = self.iteration

        self.model_param = model_param
        self.pipeline_param = pipeline_param

        with torch.no_grad():
            self.gaussians = GaussianModel(args)

            # ==== 這裡改成根據 iteration 找 ply ====
            pc_root = self.model_path / "point_cloud"

            if self.iteration <= 0:
                # 自動找最新 iteration_XXXX
                cand_dirs = sorted(pc_root.glob("iteration_*"))
                if not cand_dirs:
                    raise FileNotFoundError(f"No iteration_* folders under {pc_root}")
                pc_dir = cand_dirs[-1]
            else:
                pc_dir = pc_root / f"iteration_{self.iteration}"

            ply_file = pc_dir / "point_cloud.ply"
            print(f"[EasyRenderer] loading point cloud from: {ply_file}")
            if not ply_file.exists():
                raise FileNotFoundError(f"point_cloud.ply not found: {ply_file}")

            self.gaussians.load_ply(str(ply_file))

            bg_color = [1, 1, 1] if model_param.white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"EasyRenderer from {self.model_path} set done. (iteration={self.iteration})")

    def render(self, w2c, intrinsic, h, w, alpha_thresh=0.05):
            """
            w2c: [4,4] np.array (world→camera)
            intrinsic: [3,3] np.array
            h, w: image height, width
            alpha_thresh: 小於這個透明度的 pixel 直接當背景
            """
            with torch.no_grad():
                view = self.make_gs_view_format(w2c, intrinsic, h, w)
                rendering = render(view, self.gaussians, self.pipeline_param, self.background)

                rgb   = rendering["render"]   # [3, H, W]
                alpha = rendering["alpha"]    # [1, H, W] 或 [H, W]
                depth = rendering["depth"]

                # 確保 alpha shape 是 [1, H, W]
                if alpha.dim() == 2:
                    alpha = alpha.unsqueeze(0)

                # 建一個背景圖：跟 rgb 同 size
                bg = self.background.view(3, 1, 1).expand_as(rgb)

                # 做 alpha 閾值：低於 threshold 的 pixel 全當背景
                mask = (alpha >= alpha_thresh).float()  # [1,H,W]
                rgb_filtered = rgb * mask + bg * (1.0 - mask)

            return rgb_filtered, alpha, depth


    def make_gs_view_format(self, w2c, intrinsic, h, w):
        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]
        FovX = focal2fov(focal_length_x, w)
        FovY = focal2fov(focal_length_y, h)

        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        pseudo_cam = PseudoCamera(R=R, T=T, FoVx=FovX, FoVy=FovY, width=w, height=h)
        return pseudo_cam


if __name__ == "__main__":
    # 測試用 main，可以照舊或刪掉
    model_path = "../output/replica_dust3r_minconf1_nopseudo_nodepth_0poslr_nodensify_withpointersect_highpointersectweight/office_2/Sequence_2/"
    iteration = 10000

    easy_renderer = EasyRenderer(model_path, iteration)

    cam_file = "../utils/cam_poses.pt"
    cam = torch.load(cam_file)

    for idx in range(400, 900, 6):
        intrinsic = cam['intrinsic'][0][idx]
        c2w = cam['H_c2w'][0][idx]
        w2c = np.linalg.inv(c2w)
        h, w = cam['height_px'], cam['width_px']

        rgb, alpha, depth = easy_renderer.render(w2c, intrinsic, h, w)
        print(rgb.shape)  # [3, h, w]
        torchvision.utils.save_image(rgb, f"./{idx}.png")
