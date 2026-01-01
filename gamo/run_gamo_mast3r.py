import argparse
import copy
import json
import os
import random
import time
from glob import glob
import math
import torch
import cv2
import imagesize
import numpy as np
import torch
import trimesh
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
from tqdm import tqdm
#from depth_pro.depth_pro import create_model_and_transforms
#from depth_pro.utils import load_rgb
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from pathlib import Path


from src.modules.cam_vis import add_scene_cam
from src.modules.position_encoding import global_position_encoding_3d
from src.modules.schedulers import get_diffusion_scheduler
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview_inverse import StableDiffusionMultiViewPipeline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

import mast3r.utils.path_to_dust3r  # 確保會找到內嵌的 dust3r 子模組
from mast3r.model import AsymmetricMASt3R

# === put these at the very top ===
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1) 讓 Python 先搜到 MASt3R 的套件（裡面含 vendored dust3r & croco）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "MASt3R-SLAM", "thirdparty", "mast3r"))

# 2) 這行一定要早於所有 dust3r import
import mast3r.utils.path_to_dust3r  # 會把 vendored 的 dust3r/croco 路徑加入 sys.path


from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def np_points_padding(points):
    padding = np.ones_like(points)[..., 0:1]
    points = np.concatenate([points, padding], axis=-1)
    return points


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def save_16bit_png_depth(depth: np.ndarray, depth_png: str):
    # Ensure the numpy array's dtype is float32, then cast to float16, and finally reinterpret as uint16
    depth_uint16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)

    # Create a PIL Image from the 16-bit depth values and save it
    depth_pil = Image.fromarray(depth_uint16)

    if not depth_png.endswith(".png"):
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

    try:
        depth_pil.save(depth_png)
    except:
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError


def load_dataset(args, config, reference_cam, target_cam, reference_list, depth_list):
    ratio_set = json.load(open(f"./{args.model_dir}/ratio_set.json", "r"))
    ratio_dict = dict()
    for h, w in ratio_set:
        ratio_dict[h / w] = [h, w]
    ratio_list = list(ratio_dict.keys())

    # load dataset
    print("Loading dataset...")
    intrinsic = np.array(reference_cam["intrinsic"])
    
    """
    tar_names = list(target_cam["extrinsic"].keys())
    tar_names.sort()
    if args.target_limit is not None:
        tar_names = tar_names[:args.target_limit]
    tar_extrinsic = [np.array(target_cam["extrinsic"][k]) for k in tar_names]
    """
    

    if args.cond_num == 1:
        reference_list = [reference_list[0]]
    elif args.cond_num == 2:
        reference_list = [reference_list[0], reference_list[-1]]
    elif args.cond_num == 3:
        reference_list = reference_list[:3]
    else:
        pass

    ref_images = []
    ref_names = []
    ref_extrinsic = []
    ref_intrinsic = []
    ref_depth = []
    h, w = None, None
    for i, im in enumerate(tqdm(reference_list, desc="loading reference images")):
        img = Image.open(im).convert("RGB")
        intrinsic_ = copy.deepcopy(intrinsic)
        im = f"view{str(reference_views[i]).zfill(3)}_ref"
        if im.split("/")[-1] in reference_cam["extrinsic"]:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1]])
        else:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1].split(".")[0]])
        ref_extrinsic.append(extrinsic_)
        ref_names.append(im.split('/')[-1])

        origin_w, origin_h = img.size

        # load monocular depth
        if config.model_cfg.get("enable_depth", False):
            depth = depth_list[i]
            depth = cv2.resize(depth, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth = None
            
        #breakpoint()
        if h is None or w is None: ##Scale
            #breakpoint()
            ratio = origin_h / origin_w
            sub = [abs(ratio - r) for r in ratio_list]
            [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
            h = int(origin_h / 1) // 8 * 8
            w = int(origin_w / 1) // 8 * 8  #### 除5倍最好
            #breakpoint()   #### 3024, 4032 -> 800, 600
            print(f"[Adjusted] height: {h}, width: {w}")
            #print(f'height:{h}, width:{w}.')
        img = img.resize((w, h), Image.LANCZOS if h < origin_h else Image.BICUBIC)
        if depth is not None:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        new_w, new_h = img.size
        # rescale intrinsic
        intrinsic_[0, :] *= (new_w / reference_cam['w'])
        intrinsic_[1, :] *= (new_h / reference_cam['h'])

        img = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
        if depth is not None:
            depth = Compose([ToTensor()])(depth)

        ref_images.append(img)
        ref_intrinsic.append(intrinsic_)
        if depth is not None:
            ref_depth.append(depth)

    ref_images = torch.stack(ref_images, dim=0)
    #tar_intrinsic = [ref_intrinsic[0]] * len(tar_extrinsic)  ###直接複製conditioning image相通內參
    #tar_intrinsic = copy.deepcopy(ref_intrinsic)
    # tar_intrinsic = ref_intrinsic.clone()

    #breakpoint()
    
    ref_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_intrinsic], dim=0)
    #tar_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_intrinsic], dim=0)
    ref_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_extrinsic], dim=0)
    #tar_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_extrinsic], dim=0)
    #tar_intrinsic = ref_intrinsic.clone()

    #tar_names = copy.deepcopy(ref_names)
    tar_names = [name.replace("_ref", "") for name in ref_names]
    tar_extrinsic = ref_extrinsic.clone()
    tar_intrinsic = ref_intrinsic.clone()
    #breakpoint()
    
    # 外参t归一化
    if config.camera_longest_side is not None:
        extrinsic = torch.cat([ref_extrinsic, tar_extrinsic], dim=0)  # [N,4,4]
        c2ws = extrinsic.inverse()
        max_scale = torch.max(c2ws[:, :3, -1], dim=0)[0]
        min_scale = torch.min(c2ws[:, :3, -1], dim=0)[0]
        max_size = torch.max(max_scale - min_scale).item()
        rescale = config.camera_longest_side / max_size if max_size > config.camera_longest_side else 1.0
        ref_extrinsic[:, :3, 3:4] *= rescale
        tar_extrinsic[:, :3, 3:4] *= rescale
    else:
        rescale = 1.0

    if len(ref_depth) > 0:
        ref_depth = [r * rescale for r in ref_depth]
        ref_depth = torch.stack(ref_depth, dim=0)
    else:
        ref_depth = None



    camera_poses = {"h": h, "w": w, "intrinsic": ref_intrinsic[0].numpy().tolist(), "extrinsic": dict()}
    for i in range(len(ref_names)):
        camera_poses['extrinsic'][ref_names[i].split('.')[0].replace('_ref', '') + ".png"] = ref_extrinsic[i].numpy().tolist()
    for i in range(len(tar_names)):
        camera_poses['extrinsic'][tar_names[i].split('.')[0].replace('_ref', '') + ".png"] = tar_extrinsic[i].numpy().tolist()

    
    #breakpoint()
    return {"ref_images": ref_images, "ref_intrinsic": ref_intrinsic, "tar_intrinsic": tar_intrinsic,
            "ref_extrinsic": ref_extrinsic, "tar_extrinsic": tar_extrinsic, "ref_depth": ref_depth,
            "ref_names": ref_names, "tar_names": tar_names}


def save_depth_maps(ref_depth, output_dir="saved_depths", names=None):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(ref_depth, torch.Tensor):
        ref_depth = ref_depth.cpu().numpy()  # 轉 numpy

    N, _, H, W = ref_depth.shape

    # 計算深度值範圍（排除極端值以避免灰階過曝）
    min_val = np.percentile(ref_depth, 1)
    max_val = np.percentile(ref_depth, 99)

    for i in range(N):
        depth = ref_depth[i, 0]  # shape: [H, W]
        depth_clipped = np.clip(depth, min_val, max_val)
        depth_normalized = (depth_clipped - min_val) / (max_val - min_val + 1e-8)
        depth_img = (depth_normalized * 255).astype(np.uint8)

        # 保證檔名有 .png 副檔名
        base_name = names[i] if names is not None else f"depth_{i:02d}"
        if not base_name.lower().endswith(".png"):
            base_name += ".png"

        save_path = os.path.join(output_dir, base_name)
        plt.imsave(save_path, depth_img, cmap="gray")
        print(f"✅ Saved {save_path}")



import torch
import torch.nn.functional as F



def fov_from_K(K, h, w):
    fx, fy = K[:,0,0], K[:,1,1]
    return 2*torch.atan(w/(2*fx)), 2*torch.atan(h/(2*fy))

def cal_new_focal_scale(intrinsic, h, w, fov_x_zoom_scale, fov_y_zoom_scale):
    K = intrinsic
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]

    # 舊 FOV（rad）
    fov_x_rad_old = 2 * torch.atan(w / (2 * fx))
    fov_y_rad_old = 2 * torch.atan(h / (2 * fy))

    # 新 FOV（rad），這裡 zoom_scale 是 FOV 倍數
    fov_x_rad_new = fov_x_rad_old * fov_x_zoom_scale
    fov_y_rad_new = fov_y_rad_old * fov_y_zoom_scale

    # 計算焦距縮放比例 = tan(old/2) / tan(new/2)
    scale_x = torch.tan(fov_x_rad_old / 2) / torch.tan(fov_x_rad_new / 2)
    scale_y = torch.tan(fov_y_rad_old / 2) / torch.tan(fov_y_rad_new / 2)
    
    return scale_x, scale_y

    
    #breakpoint()
    



def fov_scale_from_size_torch(fov_old_rad, size_scale):
    """fov_old_rad: (...,) 張量（弧度）；size_scale: 標量或同形狀張量"""
    theta_old = fov_old_rad / 2
    theta_new = torch.atan(size_scale * torch.tan(theta_old))
    fov_new = 2 * theta_new
    zoom = fov_new / (fov_old_rad + 1e-8)
    return fov_new, zoom


def eval(args, config, data, pipeline, img_zoom_scale=1):
    
    N_target = data['tar_intrinsic'].shape[0]
    gen_num = config.nframe - args.cond_num
    
    #breakpoint()
    
    # save reference images
    for i in range(data['ref_images'].shape[0]):
        ref_img = ToPILImage()((data['ref_images'][i] + 1) / 2)
        ref_img.save(f"{config.save_path}/images/{data['ref_names'][i].split('.')[0]}.png")

    #warp_ref = data["warp_ref"]

    with torch.no_grad(), torch.autocast("cuda"):
        iter_times = N_target // gen_num
        if N_target % gen_num != 0:
            iter_times += 1
        for i in range(iter_times):
            print(f"synthesis target views {np.arange(N_target)[i::iter_times].tolist()}...")
            h, w = data['ref_images'].shape[2], data['ref_images'].shape[3]  ### 320, 576
            #h = h+100
            #w = w+200
            #breakpoint()
            
            gen_num_ = len(np.arange(N_target)[i::iter_times].tolist())
            print(f"Gen num {gen_num_ + args.cond_num}...")
            image = torch.cat([data["ref_images"], torch.zeros((gen_num_, 3, h, w), dtype=torch.float32)], dim=0).to("cuda") ##27
            intrinsic = torch.cat([data["ref_intrinsic"], data["tar_intrinsic"][i::iter_times]], dim=0).to("cuda")
            fovx_old, fovy_old = fov_from_K(intrinsic, h, w) ##0.9889  ##0.6801
            
            
            fovx_new, fovx_zoom_scale = fov_scale_from_size_torch(fovx_old, img_zoom_scale)
            fovy_new, fovy_zoom_scale = fov_scale_from_size_torch(fovy_old, img_zoom_scale)

            #fov_zoom_scale = 
            
            scale_x, scale_y = cal_new_focal_scale(intrinsic, h, w, fovx_zoom_scale, fovy_zoom_scale)
            
            #breakpoint()
            
            # zoom_scale =2
            
            #intrinsic[args.cond_num:, 0, 0] *= 1/zoom_scale #scale_x[args.cond_num:]
            #intrinsic[args.cond_num:, 1, 1] *= 1/zoom_scale #scale_y[args.cond_num:]
            
            #改fx, fy    ###### 放大後面的fov
            intrinsic[args.cond_num:, 0, 0] *= scale_x[args.cond_num:]
            intrinsic[args.cond_num:, 1, 1] *= scale_y[args.cond_num:]
            
            
            # ...縮放 fx, fy 後
            
            fovx_new, fovy_new = fov_from_K(intrinsic, h, w)
            print("mean FOVx old/new (deg):",
            (fovx_old.mean()*180/torch.pi).item(),
            (fovx_new.mean()*180/torch.pi).item())
            
            
            ref_old  = fovx_old[:args.cond_num]
            ref_new  = fovx_new[:args.cond_num]
            tar_old  = fovx_old[args.cond_num:]
            tar_new  = fovx_new[args.cond_num:]

            print("REF 平均 old/new:", ref_old.mean()*180/torch.pi, ref_new.mean()*180/torch.pi)
            print("TAR 平均 old/new:", tar_old.mean()*180/torch.pi, tar_new.mean()*180/torch.pi)

            #breakpoint()
            
            
            
            #     
            #breakpoint()
            
            extrinsic = torch.cat([data["ref_extrinsic"], data["tar_extrinsic"][i::iter_times]], dim=0).to("cuda")
            if data["ref_depth"] is not None:
                depth = torch.cat([data["ref_depth"], torch.zeros((gen_num_, 1, h, w), dtype=torch.float32)], dim=0).to("cuda")
            else:
                depth = None

            nframe_new = gen_num_ + args.cond_num
            config_copy = copy.deepcopy(config)
            config_copy.nframe = nframe_new
            generator = torch.Generator()
            generator = generator.manual_seed(args.seed)
            st = time.time()
            #breakpoint()
            
            tar_idx_batch = np.arange(N_target)[i::iter_times] ##[ 0,  3,  6,  9, 12, 15, 18]
            preds = pipeline(images=image, nframe=nframe_new, cond_num=args.cond_num,  #### predict novel view ## (27, 264, 480, 3)
                             key_rescale=args.key_rescale, height=h, width=w, intrinsics=intrinsic,
                             extrinsics=extrinsic, num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                             class_label=args.class_label, depth=depth, vae=pipeline.vae, generator=generator, tar_idx_batch=tar_idx_batch).images  # [f,h,w,c]
            
            
            
            
            #breakpoint()
            print("Time used:", time.time() - st)
            preds = preds[args.cond_num:]
            preds = (preds * 255).astype(np.uint8)
            #breakpoint()
            
            for j in range(preds.shape[0]):
                cv2.imwrite(f"{config.save_path}/images/{data['tar_names'][i::iter_times][j].split('.')[0]}.png", preds[j, :, :, ::-1])

            if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
                color_warps = global_position_encoding_3d(config_copy, depth, intrinsic, extrinsic,
                                                          args.cond_num, nframe=nframe_new, device=device,
                                                          pe_scale=1 / 8, embed_dim=config.model_cfg.get("coord_dim", 192),
                                                          colors=image)[0]
                #breakpoint()
                cv2.imwrite(f"{config.save_path}/warp{np.arange(N_target)[i::iter_times].tolist()}.png", color_warps[:, :, ::-1])

def save_warp_result(warped_img, valid_mask, save_dir="warped_output", index=0):
    os.makedirs(save_dir, exist_ok=True)

    # warped_img: [3, H, W] → [H, W, 3]
    warped_np = warped_img.detach().cpu().permute(1, 2, 0).numpy()
    warped_np = (warped_np + 1.0) / 2.0  # 將 [-1, 1] 映射到 [0, 1]
    warped_np = np.clip(warped_np, 0, 1)

    # valid_mask: [1, H, W] → [H, W]
    mask_np = valid_mask[0].detach().cpu().numpy()
    

    # 儲存
    plt.imsave(os.path.join(save_dir, f"warped_{index:02d}.png"), warped_np)
    plt.imsave(os.path.join(save_dir, f"mask_{index:02d}.png"), mask_np, cmap="gray")
    print(f"✅ Saved warped image and mask for index {index} to {save_dir}/")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build cam traj")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="check_points/pretrained_model", help="model directory.")
    parser.add_argument("--output_path", type=str, default="outputs/demo")
    parser.add_argument("--val_cfg", type=float, default=2.0)
    parser.add_argument("--key_rescale", type=float, default=None)
    parser.add_argument("--camera_longest_side", type=float, default=5.0)
    parser.add_argument("--nframe", type=int, default=28)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--target_limit", type=int, default=None)
    # single-view parameters
    parser.add_argument("--center_scale", type=float, default=1.0)
    parser.add_argument("--elevation", type=float, default=5.0, help="the initial elevation angle")
    parser.add_argument("--d_theta", type=float, default=0.0, help="elevation rotation angle")
    parser.add_argument("--d_phi", type=float, default=45.0, help="azimuth rotation angle")
    parser.add_argument("--d_r", type=float, default=1.0, help="the distance from camera to the world center")
    parser.add_argument("--x_offset", type=float, default=0.0, help="up moving")
    parser.add_argument("--y_offset", type=float, default=0.0, help="left moving")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--mask",type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--cam_traj", type=str, default="free",
                        choices=["free", "bi_direction", "disorder", "swing1", "swing2"])

    args = parser.parse_args()
    config = EasyDict(OmegaConf.load(os.path.join(args.model_dir, "config.yaml")))
    if config.nframe != args.nframe:
        print(f"Extend nframe from {config.nframe} to {args.nframe}.")
        config.nframe = args.nframe
        if config.nframe > 28 and args.key_rescale is None:
            args.key_rescale = 1.2
        print("key rescale", args.key_rescale)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"

    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
    else:
        input_files = glob(f"{args.input_path}/*")  # take all figures as conditional views
    args.cond_num = len(input_files)
    
    
    # 2. 推出 GT 資料夾路徑（在 input_path 的上一層）
    #    例如：.../rgb/  ->  .../GT/
    parent_dir = os.path.dirname(os.path.abspath(args.input_path))
    #gt_dir = os.path.join(parent_dir, "GT")
    render_dir = os.path.join(parent_dir, "renders")
    
    if args.mask == "hard":
        mask_dir = os.path.join(parent_dir, "masks")
    else:
        mask_dir = os.path.join(parent_dir, "masks_soft")


    # 3. 讀取 render 裡所有圖片
    if os.path.exists(render_dir):
        render_imgs = load_images(render_dir, size=512, square_ok=True)
        print(f"✅ Found {len(render_imgs)} GT images in: {render_dir}")
    else:
        render_imgs = []
        print(f"⚠️ No GT folder found at: {render_dir}")
    
    if os.path.exists(mask_dir):
        mask_imgs = load_images(mask_dir, size=512, square_ok=True)
        print(f"✅ Found {len(mask_imgs)} GT images in: {mask_dir}")
    else:
        mask_imgs = []
        print(f"⚠️ No GT folder found at: {mask_dir}")
        
           
        
    
    #breakpoint()

    ### Step1: get camera trajectory ###
    print("Get camera traj...")
    if args.cond_num > 1:
        # Masetr
        # you can put the path to a local checkpoint in model_name if needed
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        mast3r = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        mast3r.eval()
        for p in mast3r.parameters():
            p.requires_grad_(False)
        
        
        mast3r_images = load_images(input_files, size=512, square_ok=True)
        pairs = make_pairs(mast3r_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, mast3r, device, batch_size=1)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        torch.cuda.empty_cache()

        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
        c2ws = scene.get_im_poses().detach().cpu()
        cam_pos = c2ws[:, :3, -1].numpy()  # [N,3]

        max_scale = [cam_pos[:, 0].max(), cam_pos[:, 1].max(), cam_pos[:, 2].max()]
        min_scale = [cam_pos[:, 0].min(), cam_pos[:, 1].min(), cam_pos[:, 2].min()]
        cam_size = np.array(max_scale) - np.array(min_scale)
        max_size = np.max(cam_size)
        rescale = args.camera_longest_side / max_size if max_size > args.camera_longest_side else 1.0

        w2cs = torch.inverse(c2ws)
        w2cs[:, :3, 3:4] *= rescale
        c2ws[:, :3, 3:4] *= rescale
        
        Ks = scene.get_intrinsics().detach().cpu()
        origin_w, origin_h = None, None
        
        for i in range(len(input_files)):
            origin_w, origin_h = imagesize.get(input_files[i])
            new_h, new_w = mast3r_images[i]['true_shape'][0, 0], mast3r_images[i]['true_shape'][0, 1]
            Ks[i, 0] *= (origin_w / new_w)
            Ks[i, 1] *= (origin_h / new_h)
            
        #3. 深度圖
        mast3r_depths = scene.get_depthmaps()
        mast3r_depths = [d.detach().cpu() * rescale for d in mast3r_depths]
        scene.min_conf_thr = args.min_conf_thr
        
        confidence_masks = scene.get_masks()
        confidence_masks = [c.detach().cpu() for c in confidence_masks]

        # 4. 儲存 MASt3R 深度圖（灰階）
        save_dir = os.path.join(args.output_path, "mast3r_depth_viz")
        os.makedirs(save_dir, exist_ok=True)

        for i, d in enumerate(mast3r_depths):
            depth = d.numpy()
            # clip 去掉極端值，避免灰階過曝
            vmin, vmax = np.percentile(depth, 1), np.percentile(depth, 99)
            depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
            # 存成灰階圖
            plt.imsave(os.path.join(save_dir, f"depth_{i:03d}.png"), depth_norm, cmap="gray")
            print(f"✅ Saved gray depth map {i} to {save_dir}")

        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        points3d = []
        colors = []
        for i in range(len(input_files)):
            color = imgs[i]
            points = pts3d[i].detach().cpu().numpy() * rescale
            mask = confidence_masks[i].detach().cpu().numpy()
            points3d.append(points[mask])
            colors.append(color[mask])
            
        #5.彩色點雲
        points3d = np.concatenate(points3d)
        colors = np.concatenate(colors)
        colors = (np.clip(colors, 0, 1.0) * 255).astype(np.uint8)

        Ks = Ks.numpy()
        K = np.mean(Ks, axis=0)
        w2cs = w2cs.numpy()
        c2ws = c2ws.numpy()
        depth = mast3r_depths  
        depths_tensor = torch.cat(mast3r_depths, dim=0)
        print("min:", depths_tensor.min().item(), "max:", depths_tensor.max().item())

        # # === 6. 儲存彩色點雲 ===
        # save_dir = Path("Debug/pointcloud")
        # save_dir.mkdir(parents=True, exist_ok=True)
        # ply_path = save_dir / "pointcloud.ply"

        # with open(ply_path, "w") as f:
        #     f.write("ply\n")
        #     f.write("format ascii 1.0\n")
        #     f.write(f"element vertex {len(points3d)}\n")
        #     f.write("property float x\n")
        #     f.write("property float y\n")
        #     f.write("property float z\n")
        #     f.write("property uchar red\n")
        #     f.write("property uchar green\n")
        #     f.write("property uchar blue\n")
        #     f.write("end_header\n")
        #     for p, c in zip(points3d, colors):
        #         f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

        # print(f"✅ Point cloud saved to {ply_path}")

                
        # breakpoint()
        
        for i in range(len(depth)):
            depth[i][confidence_masks == False] = 0
            depth[i] = depth[i].numpy()
            depth[i] = cv2.resize(depth[i], (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        print("Build the novel frames...")
        c2ws_all = [c2ws[0]]
        w2cs_all = [w2cs[0]]
        reference_views = [0]

        nframes = []
        for i in range(len(input_files) - 1):
            if i != len(input_files) - 2:
                nframes.append((args.nframe - len(input_files)) // (len(input_files) - 1))
            else:
                nframes.append((args.nframe - len(input_files)) - int(np.sum(nframes)))

        cam_idx = 1
        for j in range(len(input_files) - 1):
            # offset interpolation
            pos0 = c2ws[j, :3, -1]
            pos1 = c2ws[j + 1, :3, -1]
            R0 = w2cs[j, :3, :3]
            R1 = w2cs[j + 1, :3, :3]
            rotation0 = Rotation.from_matrix(R0)
            rotation1 = Rotation.from_matrix(R1)
            euler_angles0 = rotation0.as_euler('xyz', degrees=True)
            euler_angles1 = rotation1.as_euler('xyz', degrees=True)

            # 检查是否有符号骤变
            sign_diff = np.sign(euler_angles0) * np.sign(euler_angles1)
            for i_ in range(len(sign_diff)):
                # 先变为连续角度180°-->360°
                if sign_diff[i_] == -1 and abs(euler_angles0[i_]) + abs(euler_angles1[i_]) > 180:
                    if euler_angles1[i_] > 0:
                        euler_angles1[i_] = -360 + euler_angles1[i_]
                    else:
                        euler_angles1[i_] = 360 + euler_angles1[i_]
            for i in range(nframes[j]):
                coef = (i + 1) / (nframes[j] + 1)
                pos_mid = (1 - coef) * pos0 + coef * pos1
                euler_angles = (1 - coef) * euler_angles0 + coef * euler_angles1
                for i_ in range(len(sign_diff)):
                    # 360°-->180°
                    if sign_diff[i_] == -1 and abs(euler_angles0[i_]) + abs(euler_angles1[i_]) > 180:
                        if euler_angles[i_] > 180:
                            euler_angles[i_] = -360 + euler_angles[i_]
                        elif euler_angles[i_] < -180:
                            euler_angles[i_] = 360 + euler_angles[i_]
                print(j, euler_angles0, euler_angles1, euler_angles)
                # 将欧拉角转换回旋转矩阵
                rotation_from_euler = Rotation.from_euler('xyz', euler_angles, degrees=True)
                R_mid = rotation_from_euler.as_matrix()
                R_mid = R_mid.T
                c2w_mid = np.concatenate([R_mid.reshape((3, 3)), pos_mid.reshape((3, 1))], axis=-1)
                c2w_mid = np.concatenate([c2w_mid, np.zeros((1, 4))], axis=0)
                c2w_mid[-1, -1] = 1
                w2c_mid = np.linalg.inv(c2w_mid)

                c2ws_all.append(c2w_mid)
                w2cs_all.append(w2c_mid)

                cam_idx += 1

            c2ws_all.append(c2ws[j + 1])
            w2cs_all.append(w2cs[j + 1])
            reference_views.append(len(c2ws_all) - 1)

        print("Multi-view trajectory building over...")


        #breakpoint()
                
    

    # save pointcloud and cameras
    scene = trimesh.Scene()
    for i in range(len(c2ws_all)):
        add_scene_cam(scene, c2ws_all[i], CAM_COLORS[i % len(CAM_COLORS)], None, imsize=(512, 512), screen_width=0.03)

    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{save_path}/pcd.ply")
    scene.export(file_obj=f"{save_path}/cameras.glb")

    reference_cam = {"h": origin_h, "w": origin_w, "intrinsic": K.tolist()}
    reference_cam["extrinsic"] = dict()
    target_cam = copy.deepcopy(reference_cam)
    
    #breakpoint() ##origin_h 3024
    
    for i in range(len(reference_views)):
        reference_cam["extrinsic"][f"view{str(reference_views[i]).zfill(3)}_ref"] = w2cs_all[reference_views[i]].tolist()

    for i in range(len(w2cs_all)):
        if i not in reference_views:
            target_cam["extrinsic"][f"view{str(i).zfill(3)}"] = w2cs_all[i].tolist()

    ### Step2: generate multi-view images ###
    # init model
    print("load model...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path,
                                        subfolder="vae", local_files_only=True)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=0,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True,
                                                local_files_only=True)
    unet.requires_grad_(False)
    # load pretained weights
    weights = torch.load(f"{args.model_dir}/ema_unet.pt", map_location="cpu")
    unet.load_state_dict(weights)
    unet.eval()

    weight_dtype = torch.float16
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    img_zoom_scale = 1 / 0.6 #0.8/0.6 #1/0.6 #2 #1/0.6#2 #1/0.6 #4/3#2 #4/3 #2 #2 #1
    
    scheduler = get_diffusion_scheduler(config, name="DDIM")
    ### Get novel view
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True,
        zoom_scale=1/img_zoom_scale, 
        render_imgs = render_imgs,
        mask_imgs = mask_imgs,
        mask_mode = args.mask
        #warp_ref = warp_ref
    )
    pipeline = pipeline.to(device)

    # load dataset
    args.dataset_dir = save_path
    config.save_path = save_path
    data = load_dataset(args, config, reference_cam, target_cam, input_files, depth)  ###LOAD_DATASET
    
    """
    "ref_images": ref_images, "ref_intrinsic": ref_intrinsic, "tar_intrinsic": tar_intrinsic,
            "ref_extrinsic": ref_extrinsic, "tar_extrinsic": tar_extrinsic, "ref_depth": ref_depth,
            "ref_names": ref_names, "tar_names": tar_names
    """
    
    
    
    # ref_img : [20, 3, 408, 616]
    # ref_intrinsic : [20, 3, 3]
    # ref_depth : [20, 1, 408, 616]
    save_depth_maps(data["ref_depth"], output_dir="ref_depth_viz_duster", names=data["ref_names"])
    
    #warp_ref_img(data, zoom_scale)
    #breakpoint()

    os.makedirs(f"{save_path}/images", exist_ok=True)
    eval(args, config, data, pipeline, img_zoom_scale)

    results = glob(f"{config.save_path}/images/view*.png")
    
    results = [r for r in results if "_ref" not in r]

    results.sort(key=lambda x: int(x.split('/')[-1].replace(".png", "").replace("view", "")))
    
    #results.sort(key=lambda x: int(x.split('/')[-1].replace(".png", "").replace("view", "").replace("_ref", "")))
    
    clip = ImageSequenceClip(results, fps=15)
    clip.write_videofile(f"{config.save_path}/output.mp4", fps=15)
