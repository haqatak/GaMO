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
import json
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
"""
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
"""



import time

def prepare_output_and_logger(args):    
    if not args.model_path:
        # ➔ 加入 timestamp 與 scene name
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # ➔ 取得 scene name (例如從 args.source_path 的最後一層資料夾名稱)
        #scene_name = os.path.basename(args.source_path.rstrip("/"))
        path_parts = args.source_path.rstrip("/").split("/")
        
        """
        if len(path_parts) >= 3:
            # 前兩層資料夾前四字母
            prev2 = path_parts[-3][:7]
            prev1 = path_parts[-2][:7]
            # 最後一層資料夾前四字母
            last = path_parts[-1][:4]

            scene_name = prev2 + "/" + prev1 + "/" + last
        else:
            # fallback: 直接使用最後一層
            scene_name = path_parts[-1][:4]
        """
        
        if len(path_parts) >= 3:
            # 前兩層資料夾完整名稱
            prev2 = path_parts[-3]
            prev1 = path_parts[-2]
            # 最後一層資料夾完整名稱
            last = path_parts[-1]

            scene_name = os.path.join(prev2, prev1, last)
        else:
            # fallback: 直接使用最後一層
            scene_name = path_parts[-1]
        
        
        # ➔ 合成 output folder name
        output_name = f"{scene_name}/{timestamp}"

        args.model_path = os.path.join("./output/", output_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, train_test_exp):

    import torch.nn.functional as F  # 用於 interpolate

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # === 初始化 LPIPS (VGG & Alex) 一次並快取 ===
    if not hasattr(training_report, "_lpips_inited"):
        training_report._lpips_inited = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_report._lpips_vgg = None
        training_report._lpips_alex = None
        try:
            from lpips import LPIPS
            training_report._lpips_vgg  = LPIPS(net='vgg').to(device).eval()
            for p in training_report._lpips_vgg.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[Val] LPIPS(VGG) init failed: {e}. Skip LPIPS(VGG).")
        try:
            from lpips import LPIPS
            training_report._lpips_alex = LPIPS(net='alex').to(device).eval()
            for p in training_report._lpips_alex.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[Val] LPIPS(Alex) init failed: {e}. Skip LPIPS(Alex).")

    def _lpips_score(model, img, gt):
        """img, gt: [C,H,W], in [0,1]. 會自動 resize 到 gt 尺寸並轉 [-1,1] 再算 LPIPS。"""
        if model is None:
            return None
        if img.shape[-2:] != gt.shape[-2:]:
            img = F.interpolate(img.unsqueeze(0), size=gt.shape[-2:], mode='bilinear', align_corners=False)[0]
        img = img.unsqueeze(0) * 2.0 - 1.0
        gt  = gt.unsqueeze(0)  * 2.0 - 1.0
        with torch.no_grad():
            s = model(img, gt)
        return s.mean().double()

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            cams = config['cameras']
            if cams and len(cams) > 0:
                l1_avg = 0.0
                psnr_avg = 0.0
                lpips_vgg_avg = 0.0
                lpips_alex_avg = 0.0
                n = len(cams)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    for idx, viewpoint in enumerate(cams):
                        # render
                        image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0
                        ).to(device)
                        gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)

                        # half-exposure crop if needed
                        if train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                        # TB dump
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/render",
                                image[None], global_step=iteration
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                    gt_image[None], global_step=iteration
                                )

                        # metrics
                        l1_avg   += l1_loss(image, gt_image).mean().double()
                        psnr_avg += psnr(image, gt_image).mean().double()

                        # LPIPS (VGG / Alex)
                        s_vgg  = _lpips_score(getattr(training_report, "_lpips_vgg", None),  image, gt_image)
                        s_alex = _lpips_score(getattr(training_report, "_lpips_alex", None), image, gt_image)
                        if s_vgg  is not None: lpips_vgg_avg  += s_vgg
                        if s_alex is not None: lpips_alex_avg += s_alex

                # 平均
                l1_avg   /= n
                psnr_avg /= n
                have_vgg  = getattr(training_report, "_lpips_vgg", None)  is not None
                have_alex = getattr(training_report, "_lpips_alex", None) is not None
                if have_vgg:  lpips_vgg_avg  /= n
                if have_alex: lpips_alex_avg /= n

                # 印出
                line = f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_avg} PSNR {psnr_avg}"
                if have_vgg:  line += f" LPIPS(VGG) {lpips_vgg_avg}"
                if have_alex: line += f" LPIPS(Alex) {lpips_alex_avg}"
                print(line)

                # TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_avg, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_avg, iteration)
                    if have_vgg:
                        tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - lpips_vgg", lpips_vgg_avg, iteration)
                    if have_alex:
                        tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - lpips_alex", lpips_alex_avg, iteration)

                # 寫入 result.json
                result_path = os.path.join(scene.model_path, "result.json")
                if os.path.exists(result_path):
                    try:
                        with open(result_path, "r") as f:
                            results = json.load(f)
                    except json.JSONDecodeError:
                        results = {}
                else:
                    results = {}

                results[str(iteration)] = results.get(str(iteration), {})
                block = {
                    "L1": float(l1_avg),
                    "PSNR": float(psnr_avg),
                }
                if have_vgg:
                    block["LPIPS_VGG"] = float(lpips_vgg_avg)
                if have_alex:
                    block["LPIPS_ALEX"] = float(lpips_alex_avg)

                results[str(iteration)][config['name']] = block

                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 1_0000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 1_0000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
