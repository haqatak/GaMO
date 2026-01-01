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
import glob
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.general_utils import chamfer_dist
import numpy as np
import json
import cv2
import math
import torch
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from scene.pcd2img import project_point_cloud_to_image
import re


def extract_number(filename):
    """
    Extract the number from the filename using regex.
    """
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else None


class CameraInfo(NamedTuple):
    fid: int
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array
    projected_image: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_indices: list
    train_cameras: list
    test_cameras: list
    all_cameras: list
    project_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}



def _parse_dual_images_and_sparse(images):
    """
    支援 "--images trainDir:sparseTag | testDir:sparseTag" 的格式。
    例：'rgb:0|rgb_test:test' -> ('rgb','0','rgb_test','test')
    如果不是 dual 格式，回傳 None。
    """
    if images is None:
        return None
    parts = images.split('|')
    if len(parts) != 2:
        return None
    def _split_one(p):
        p = p.strip()
        if ':' in p:
            a,b = p.split(':',1)
            return a.strip(), b.strip()
        else:
            # 只有影像資料夾名，預設 sparse tag 用 '0'
            return p.strip(), '0'
    train_img, train_sparse = _split_one(parts[0])
    test_img,  test_sparse  = _split_one(parts[1])
    return train_img, train_sparse, test_img, test_sparse



def readColmapCamerasReplica(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping, projected_dir=None):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys(), key=extract_number)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        try: 
            bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]
        except: 
            bounds = np.array([1., 10.])

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        
        with Image.open(rgb_path) as im:
            # 只要把像素讀進記憶體就好，避免持有檔案 handle
            image = im.convert("RGB").copy()

        projected_image = None
        mask = None
        if projected_dir is not None:
            proj_path = os.path.join(projected_dir, rgb_name + ".png")
            mask_path = os.path.join(projected_dir, rgb_name + "_mask.png")
            if os.path.exists(proj_path):
                with Image.open(proj_path) as im:
                    projected_image = im.copy()
            if os.path.exists(mask_path):
                with Image.open(mask_path) as im:
                    mask = (np.array(im) / 255.).astype(np.float32)
        # print("--", extr.name, image_path, image_name, rgb_path, rgb_name)

        cam_info = CameraInfo(fid=0, uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=rgb_path,
                image_name=rgb_name, width=width, height=height, mask=mask, bounds=bounds, projected_image=projected_image)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasScannetpp(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys(), key=extract_number)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        try: 
            bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]
        except: 
            bounds = np.array([1., 10.])

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        # print("--", extr.name, image_path, image_name, rgb_path, rgb_name)
        
        projected_image, mask = None, None
        cam_info = CameraInfo(fid=0, uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=rgb_path,
                image_name=rgb_name, width=width, height=height, mask=mask, bounds=bounds, projected_image=projected_image)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def farthest_point_sampling(points, k):
    """
    Sample k points from input pointcloud data points using Farthest Point Sampling.

    Parameters:
    points: numpy.ndarray
        The input pointcloud data, a numpy array of shape (N, D) where N is the
        number of points and D is the dimensionality of each point.
    k: int
        The number of points to sample.

    Returns:
    sampled_points: numpy.ndarray
        The sampled pointcloud data, a numpy array of shape (k, D).
    """
    N, D = points.shape
    farthest_pts = np.zeros((k, D))
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(k):
        farthest_pts[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return farthest_pts


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, dataset, eval, n_views=0, 
                        dust3r_min_conf_thr=1, demo_setting=False, 
                        get_dust3r_pcd=False, replica_use_project_cam=False):
    
    scene = path.split("/")[-1]
    
    dual_spec = _parse_dual_images_and_sparse(images)

    def _load_split(split_images_dir, split_sparse_tag):
        """讀一個 split（指定 images 資料夾與 sparse 子目錄標籤），回傳 cam_infos（已排序）"""
        # 讀這個 split 的 cameras / images（支援 bin / txt）
        try:
            cams_intr_file = os.path.join(path, f"sparse/{split_sparse_tag}", "cameras.bin")
            imgs_extr_file = os.path.join(path, f"sparse/{split_sparse_tag}", "images.bin")
            _extr = read_extrinsics_binary(imgs_extr_file)
            _intr = read_intrinsics_binary(cams_intr_file)
        except:
            imgs_extr_file = os.path.join(path, f"sparse/{split_sparse_tag}", "images.txt")
            cams_intr_file = os.path.join(path, f"sparse/{split_sparse_tag}", "cameras.txt")
            _extr = read_extrinsics_text(imgs_extr_file)
            _intr = read_intrinsics_text(cams_intr_file)

        _extr = {_extr[k].name: _extr[k] for k in _extr}

        # 蒐集影像清單
        mapping = [f for f in sorted(glob.glob(os.path.join(path, split_images_dir, '*')), key=extract_number)
                if f.lower().endswith(('jpg','jpeg','png'))]

        # 依資料集型別建立 cam_infos
        if dataset == "Replica" or dataset == "Replica_crop" or dataset == "Replica_6+6":
            _projected_dir = os.path.join("./projected_dir", scene)
            _unsorted = readColmapCamerasReplica(_extr, _intr,
                        images_folder=os.path.join(path, split_images_dir), path=path,
                        rgb_mapping=mapping, projected_dir=_projected_dir)
            
        elif dataset in ("Scannetpp", "re10k"):
            _unsorted = readColmapCamerasScannetpp(_extr, _intr,
                        images_folder=os.path.join(path, split_images_dir), path=path,
                        rgb_mapping=mapping)
        else:
            _unsorted = readColmapCamerasReplica(_extr, _intr,
                        images_folder=os.path.join(path, split_images_dir), path=path,
                        rgb_mapping=mapping)

        _cams = sorted(_unsorted.copy(), key=lambda x: extract_number(x.image_name))
        return _cams


    if dataset == "Replica": 
        split = path.split("/")
        scene, seq = split[-2], split[-1]
        scene, scene_id = scene.split("_")[0], scene.split("_")[1]
        seq_id = seq.split("_")[1]
        scene = scene+scene_id+"_seq"+seq_id
        
        if not demo_setting: 
            ply_path = os.path.join("./dust3r_results/Replica_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(n_views, dust3r_min_conf_thr, scene))
        else: 
            print("=> demo setting. ")
            ply_path = os.path.join("./dust3r_results/Replica_6v_thr{}_trimeshsave_minconf1_demosetting/{}/sparse/0/points3D.ply".format(dust3r_min_conf_thr, scene))
   
    elif dataset == "Replica_crop":
        split = path.split("/")
        scene, seq = split[-2], split[-1]
        scene, scene_id = scene.split("_")[0], scene.split("_")[1]
        seq_id = seq.split("_")[1]
        scene = scene+scene_id+"_seq"+seq_id
        
        if not demo_setting: 
            ply_path = os.path.join("./dust3r_results/Replica_crop_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(n_views, dust3r_min_conf_thr, scene))
        else: 
            print("=> demo setting. ")
            ply_path = os.path.join("./dust3r_results/Replica_crop_6v_thr{}_trimeshsave_minconf1_demosetting/{}/sparse/0/points3D.ply".format(dust3r_min_conf_thr, scene))
   
    elif dataset == "Replica_6+6":
        #breakpoint()
        split = path.split("/")
        scene, seq = split[-2], split[-1]
        scene, scene_id = scene.split("_")[0], scene.split("_")[1]
        seq_id = seq.split("_")[1]
        scene = scene+scene_id+"_seq"+seq_id
        
        if not demo_setting: 
            ply_path = os.path.join("./dust3r_results/Replica_6+6_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(n_views, dust3r_min_conf_thr, scene))
        else: 
            print("=> demo setting. ")
            ply_path = os.path.join("./dust3r_results/Replica_6+6_6v_thr{}_trimeshsave_minconf1_demosetting/{}/sparse/0/points3D.ply".format(dust3r_min_conf_thr, scene))
   
         
    elif dataset == "Scannetpp": 
        # path: './data/ScanNetpp/8a20d62ac0'
        scene_id = path.split("/")[-1]
        scene = scene_id
        
        # only support 6 views for scannetpp
        if not demo_setting: 
            ply_path = os.path.join("./dust3r_results/Scannetpp_6v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(dust3r_min_conf_thr, scene))
        else: 
            print("=> demo setting. ")
            ply_path = os.path.join("./dust3r_results/Scannetpp_6v_thr{}_trimeshsave_minconf1_demosetting/{}/sparse/0/points3D.ply".format(dust3r_min_conf_thr, scene))
    
    elif dataset == "re10k": 
        ply_path = os.path.join("./dust3r_results/re10k_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(n_views, dust3r_min_conf_thr, scene))

    
    else:
        # 不拆 path，直接拿最後一層當 scene 名
        scene = os.path.basename(path.rstrip("/"))

        if not demo_setting:
            ply_path = os.path.join(
                "./dust3r_results/Replica_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(
                    n_views, dust3r_min_conf_thr, scene
                )
            )
        else:
            print("=> demo setting.")
            ply_path = os.path.join(
                "./dust3r_results/Replica_{}v_thr{}_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(
                    n_views, dust3r_min_conf_thr, scene
                )
            )
            
            
    print(f"=> {n_views} views training. ply_path: {ply_path}. ")

    # ========= 若啟用雙資料夾模式，直接各讀一份，並跳過固定 index split =========
    if dual_spec is not None:
        train_img_dir, train_sparse_tag, test_img_dir, test_sparse_tag = dual_spec

        # 讀 train / test 兩份
        train_all = _load_split(train_img_dir, train_sparse_tag)  # list[CameraInfo]
        test_all  = _load_split(test_img_dir,  test_sparse_tag)   # list[CameraInfo]
        #test_all = _load_split(test_img_dir, train_sparse_tag)  # list[CameraInfo]
        ### 測試中

        # 以影像檔名的數字作為全域 index（確保檔名如 rgb_68.png -> 68）
        def idx_map(cams):
            m = {}
            for c in cams:
                k = extract_number(c.image_name)
                if k is not None:
                    m[k] = c
            return m

        train_map = idx_map(train_all)
        test_map  = idx_map(test_all)
        #breakpoint()
        # -------- Replica: 決定固定訓練 indices ----------
        if dataset == "Replica" or dataset == "Replica_crop":
            split = path.split("/")
            scene_name, seq = split[-2], split[-1]
            scene_root, scene_id = scene_name.split("_")[0], scene_name.split("_")[1]
            seq_id = seq.split("_")[1]
            scene_key = f"{scene_root}{scene_id}_seq{seq_id}"

            # 與原版一致的 index 表
            TRAIN_IDX_FIXED = {
                "office2_seq2": [244, 291, 436, 607, 760, 831], 
                "office3_seq1": [22, 98, 315, 504, 581, 731], 
                "office4_seq2": [233, 305, 440, 555, 759, 806], 
                "room0_seq2":   [5, 80, 187, 392, 497, 658], 
                "room1_seq1":   [17, 39, 125, 349, 449, 840], 
                "room2_seq1":   [61, 178, 323, 485, 526, 758], 
            }
            TRAIN_IDX_9v = {
                "office2_seq2": [159, 244, 291, 436, 510, 607, 684, 760, 831], 
                "office3_seq1": [22, 98, 174, 264, 315, 504, 581, 633, 731], 
                "office4_seq2": [49, 171, 233, 305, 440, 555, 655, 759, 806], 
                "room0_seq2":   [5, 80, 187, 296, 392, 497, 548, 658, 723], 
                "room1_seq1":   [17, 39, 125, 251, 349, 449, 542, 656, 840], 
                "room2_seq1":   [61, 178, 270, 323, 400, 485, 526, 601, 758], 
            }
            TRAIN_IDX_3v = {
                "office2_seq2": [244, 291, 436], 
                "office3_seq1": [22, 98, 315], 
                "office4_seq2": [233, 305, 440], 
                "room0_seq2":   [392, 497, 658], 
                "room1_seq1":   [17, 39, 125], 
                "room2_seq1":   [323, 485, 526], 
            }

            if n_views == 6:
                TRAIN_IDX = TRAIN_IDX_FIXED
            elif n_views == 9:
                TRAIN_IDX = TRAIN_IDX_9v
            elif n_views == 3:
                TRAIN_IDX = TRAIN_IDX_3v
            else:
                raise ValueError(f"Unsupported n_views={n_views} for Replica dual mode.")

            if scene_key not in TRAIN_IDX:
                raise KeyError(f"No TRAIN_IDX for scene '{scene_key}'.")

            train_indices_fixed = TRAIN_IDX[scene_key]  # 例如 [5,80,187,392,497,658]

            # 用 fixed indices 從 train_map 擷取實際 camera
            # 若 train_dir 不含其中某些 index，就跳過並印出警告
            picked_train = []
            used_train_indices = []
            for idx in train_indices_fixed:
                cam = train_map.get(idx, None)
                if cam is not None:
                    picked_train.append(cam)
                    used_train_indices.append(idx)
                else:
                    print(f"[warn] train set missing frame index {idx}; skipped.")

            if len(picked_train) == 0:
                raise RuntimeError("No training cameras found in dual train set based on fixed indices.")

            # 產生 test indices：對每個 train idx 取 ±50 區間（不含本身），步長 10
            # 然後只保留 test_map 存在的那些 index
            test_idx_candidates = set()
            for t in used_train_indices:
                left  = max(0, t - 50)
                right = t + 50
                # 去掉本身 t，步長 10
                local = list(range(left, t)) + list(range(t + 1, right + 1))
                local = local[::10]
                for k in local:
                    if k in test_map:
                        test_idx_candidates.add(k)

            picked_test = [test_map[k] for k in sorted(test_idx_candidates)]

            # 指派 fid（train 依 0..N-1）
            for i in range(len(picked_train)):
                picked_train[i] = picked_train[i]._replace(fid=i)

            # all_cameras 可以給 union（只供 debug / 非必要）
            all_union = picked_train + picked_test

            pcd = fetchPly(ply_path) if ply_path is not None else None
            nerf_normalization = getNerfppNorm(picked_train)

            scene_info = SceneInfo(
                point_cloud        = pcd,
                train_indices      = used_train_indices,  # 回報實際用到的 global indices
                train_cameras      = picked_train,
                test_cameras       = picked_test,
                all_cameras        = all_union,
                project_cameras    = None,
                nerf_normalization = nerf_normalization,
                ply_path           = ply_path
            )
            #breakpoint()
            return scene_info
        
        elif dataset == "Replica_6+6":
            #breakpoint()
            split = path.split("/")
            scene_name, seq = split[-2], split[-1]
            scene_root, scene_id = scene_name.split("_")[0], scene_name.split("_")[1]
            seq_id = seq.split("_")[1]
            scene_key = f"{scene_root}{scene_id}_seq{seq_id}"
            """
            # 與原版一致的 index 表
            TRAIN_IDX_FIXED = {
                "office2_seq2": [244, 291, 436, 607, 760, 831, 24400, 29100, 43600, 60700, 76000, 83100], 
                "office3_seq1": [22, 98, 315, 504, 581, 731], 
                "office4_seq2": [233, 305, 440, 555, 759, 806], 
                "room0_seq2":   [5, 80, 187, 392, 497, 658], 
                "room1_seq1":   [17, 39, 125, 349, 449, 840], 
                "room2_seq1":   [61, 178, 323, 485, 526, 758], 
            }
            TRAIN_IDX_9v = {
                "office2_seq2": [159, 244, 291, 436, 510, 607, 684, 760, 831], 
                "office3_seq1": [22, 98, 174, 264, 315, 504, 581, 633, 731], 
                "office4_seq2": [49, 171, 233, 305, 440, 555, 655, 759, 806], 
                "room0_seq2":   [5, 80, 187, 296, 392, 497, 548, 658, 723], 
                "room1_seq1":   [17, 39, 125, 251, 349, 449, 542, 656, 840], 
                "room2_seq1":   [61, 178, 270, 323, 400, 485, 526, 601, 758], 
            }
            TRAIN_IDX_3v = {
                "office2_seq2": [244, 291, 436], 
                "office3_seq1": [22, 98, 315], 
                "office4_seq2": [233, 305, 440], 
                "room0_seq2":   [392, 497, 658], 
                "room1_seq1":   [17, 39, 125], 
                "room2_seq1":   [323, 485, 526], 
            }

            if n_views == 6:
                TRAIN_IDX = TRAIN_IDX_FIXED
            elif n_views == 9:
                TRAIN_IDX = TRAIN_IDX_9v
            elif n_views == 3:
                TRAIN_IDX = TRAIN_IDX_3v
            else:
                raise ValueError(f"Unsupported n_views={n_views} for Replica dual mode.")

            if scene_key not in TRAIN_IDX:
                raise KeyError(f"No TRAIN_IDX for scene '{scene_key}'.")

            """
            TRAIN_IDX = {
                "office2_seq2": [244, 291, 436, 607, 760, 831, 24400, 29100, 43600, 60700, 76000, 83100], 
                "office3_seq1": [22, 98, 315, 504, 581, 731], 
                "office4_seq2": [233, 305, 440, 555, 759, 806], 
                "room0_seq2":   [5, 80, 187, 392, 497, 658], 
                "room1_seq1":   [17, 39, 125, 349, 449, 840], 
                "room2_seq1":   [61, 178, 323, 485, 526, 758], 
            }
            
            
            train_indices_fixed = TRAIN_IDX[scene_key]  # 例如 [5,80,187,392,497,658]
            breakpoint()
            # 用 fixed indices 從 train_map 擷取實際 camera
            # 若 train_dir 不含其中某些 index，就跳過並印出警告
            picked_train = []
            used_train_indices = []
            for idx in train_indices_fixed:
                cam = train_map.get(idx, None)
                if cam is not None:
                    picked_train.append(cam)
                    used_train_indices.append(idx)
                else:
                    print(f"[warn] train set missing frame index {idx}; skipped.")

            if len(picked_train) == 0:
                raise RuntimeError("No training cameras found in dual train set based on fixed indices.")

            # 產生 test indices：對每個 train idx 取 ±50 區間（不含本身），步長 10
            # 然後只保留 test_map 存在的那些 index
            test_idx_candidates = set()
            for t in used_train_indices:
                left  = max(0, t - 50)
                right = t + 50
                # 去掉本身 t，步長 10
                local = list(range(left, t)) + list(range(t + 1, right + 1))
                local = local[::10]
                for k in local:
                    if k in test_map:
                        test_idx_candidates.add(k)

            picked_test = [test_map[k] for k in sorted(test_idx_candidates)]

            # 指派 fid（train 依 0..N-1）
            for i in range(len(picked_train)):
                picked_train[i] = picked_train[i]._replace(fid=i)

            # all_cameras 可以給 union（只供 debug / 非必要）
            all_union = picked_train + picked_test

            pcd = fetchPly(ply_path) if ply_path is not None else None
            nerf_normalization = getNerfppNorm(picked_train)

            scene_info = SceneInfo(
                point_cloud        = pcd,
                train_indices      = used_train_indices,  # 回報實際用到的 global indices
                train_cameras      = picked_train,
                test_cameras       = picked_test,
                all_cameras        = all_union,
                project_cameras    = None,
                nerf_normalization = nerf_normalization,
                ply_path           = ply_path
            )
            
            return scene_info

        else: #### Scene dual
            #breakpoint()
            # 非 Replica 的 dual 模式：這裡你可依需要模仿 Scannetpp/re10k 的規則
            # 先簡單地把整個 train_all / test_all 回傳
            for i in range(len(train_all)):
                train_all[i] = train_all[i]._replace(fid=i)
            pcd = fetchPly(ply_path) if ply_path is not None else None
            nerf_normalization = getNerfppNorm(train_all)
            return SceneInfo(
                point_cloud        = pcd,
                train_indices      = list(range(len(train_all))),
                train_cameras      = train_all,
                test_cameras       = test_all,
                all_cameras        = (train_all + test_all),
                project_cameras    = None,
                nerf_normalization = nerf_normalization,
                ply_path           = ply_path
            )
            

    # ========= 否則走舊邏輯（單資料夾 + 固定 index split） =========
    #breakpoint()
    
    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    print("Read camera files: ", cameras_intrinsic_file, " ", cameras_extrinsic_file)
    
    reading_dir = "images" if images == None else images
    print("=> reading dir: ", reading_dir)
    
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    
    if dataset == "Replica" or dataset == "Scannetpp" or dataset == "re10k" or dataset == "Replica_crop" or dataset == "Replica_6+6": 
        rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')), key=extract_number)
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] 
    else: 
        rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')), key=extract_number)
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] 
        #raise NotImplementedError
    
    if dataset == "Replica" or "Replica_crop" or "Replica_6+6": 
        projected_dir = os.path.join("./projected_dir", scene)
        cam_infos_unsorted = readColmapCamerasReplica(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir), path=path, 
                             rgb_mapping=rgb_mapping, projected_dir=projected_dir)
    elif dataset == "Scannetpp" or dataset == "re10k": 
        cam_infos_unsorted = readColmapCamerasScannetpp(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir), path=path, 
                             rgb_mapping=rgb_mapping)
    else : 
        projected_dir = os.path.join("./projected_dir", scene)
        cam_infos_unsorted = readColmapCamerasReplica(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir), path=path, 
                             rgb_mapping=rgb_mapping, projected_dir=projected_dir)
    #breakpoint()
    # print("before sorted: ")
    # for cam_info in cam_infos_unsorted:
    #     print(cam_info.image_name)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : extract_number(x.image_name))
    # print("**after sorted: ")
    # for cam_info in cam_infos:
    #     print(cam_info.image_name)

    
    if eval:
        print("Dataset Type: ", dataset)
        if dataset == "Replica" or dataset == "Replica_crop": 
            if not demo_setting: 
                TRAIN_IDX_FIXED = {
                    "office2_seq2": [244, 291, 436, 607, 760, 831], 
                    "office3_seq1": [22, 98, 315, 504, 581, 731], 
                    "office4_seq2": [233, 305, 440, 555, 759, 806], 
                    "room0_seq2": [5, 80, 187, 392, 497, 658], 
                    "room1_seq1": [17, 39, 125, 349, 449, 840], 
                    "room2_seq1": [61, 178, 323, 485, 526, 758], 
                }

                # 9 view
                TRAIN_IDX_9v = {
                    "office2_seq2": [159, 244, 291, 436, 510, 607, 684, 760, 831], 
                    "office3_seq1": [22, 98, 174, 264, 315, 504, 581, 633, 731], 
                    "office4_seq2": [49, 171, 233, 305, 440, 555, 655, 759, 806], 
                    "room0_seq2": [5, 80, 187, 296, 392, 497, 548, 658, 723], 
                    "room1_seq1": [17, 39, 125, 251, 349, 449, 542, 656, 840], 
                    "room2_seq1": [61, 178, 270, 323, 400, 485, 526, 601, 758], 
                }


                # 3 view, cannot cover the whole scene, can only cover a partial
                TRAIN_IDX_3v = {
                    "office2_seq2": [244, 291, 436], 
                    "office3_seq1": [22, 98, 315], 
                    "office4_seq2": [233, 305, 440], 
                    "room0_seq2": [392, 497, 658], 
                    "room1_seq1": [17, 39, 125], 
                    "room2_seq1": [323, 485, 526], 
                }

                if n_views == 6: 
                    TRAIN_IDX = TRAIN_IDX_FIXED
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX_FIXED
                    print(f"=> Replica 6 views. train_idx: {TRAIN_IDX}. ")
                
                # replica 3/9 views settings. corresponding to Table A1
                elif n_views == 9: 
                    TRAIN_IDX = TRAIN_IDX_9v
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX_FIXED
                    # testing views for 6/9 views are identical
                    print(f"=> Replica 9 views. train_idx: {TRAIN_IDX}. ")
                elif n_views == 3: 
                    TRAIN_IDX = TRAIN_IDX_3v
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX
                    # for 3 view, only cover partial scene
                    print(f"=> Replica 3 views. train_idx: {TRAIN_IDX}. ")
            else: 

                # demo setting, not used commonly. only for better visualization on the project page. 
                TRAIN_IDX = {
                    "office2_seq2": [244, 291, 436, 574, 760, 831], 
                    "office3_seq1": [22, 98, 187, 315, 504, 581], 
                    "room0_seq2": [80, 187, 392, 497, 658, 833], 
                    "office4_seq1": [0, 242, 370, 401, 554, 822],
                    "room1_seq1": [17, 39, 125, 349, 449, 840], 
                    "room2_seq1": [61, 178, 323, 485, 526, 758], 
                }
                TRAIN_IDX_FOR_TEST = TRAIN_IDX
            
            split = path.split("/")
            scene, seq = split[-2], split[-1]
            scene, scene_id = scene.split("_")[0], scene.split("_")[1]
            seq_id = seq.split("_")[1]
            train_idx = TRAIN_IDX[scene+scene_id+"_seq"+seq_id]
            train_idx_for_test_idx_generate = TRAIN_IDX_FOR_TEST[scene+scene_id+"_seq"+seq_id]
            scene = scene+scene_id+"_seq"+seq_id

            test_idx = []
            for idx in train_idx_for_test_idx_generate: 
                range_idx_left = [i for i in range(max(0, idx-50), idx)]
                range_idx_right = [i for i in range(idx+1, min(idx+50, len(cam_infos)))]
                range_idx = range_idx_left + range_idx_right
                range_idx = range_idx[::10] # NOTE
                test_idx.extend(range_idx)
            test_idx = list(set(test_idx))
            test_idx.sort()
            
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
            
            if replica_use_project_cam: 
                gap = 6
                project_cam_infos = [cam_infos[idx] for idx in range(len(cam_infos))[::gap]]
                # NOTE: we only use the project cam to train baseline
            else: 
                project_cam_infos = None
        
        elif dataset == "Replica_6+6": 
            #breakpoint()
            """
            if not demo_setting: 
                
                
                TRAIN_IDX_FIXED = {
                    "office2_seq2": [244, 291, 436, 607, 760, 831, 24400, 29100, 43600, 60700, 76000, 83100], 
                    "office3_seq1": [22, 98, 315, 504, 581, 731], 
                    "office4_seq2": [233, 305, 440, 555, 759, 806], 
                    "room0_seq2": [5, 80, 187, 392, 497, 658], 
                    "room1_seq1": [17, 39, 125, 349, 449, 840], 
                    "room2_seq1": [61, 178, 323, 485, 526, 758], 
                }

                # 9 view
                TRAIN_IDX_9v = {
                    "office2_seq2": [159, 244, 291, 436, 510, 607, 684, 760, 831], 
                    "office3_seq1": [22, 98, 174, 264, 315, 504, 581, 633, 731], 
                    "office4_seq2": [49, 171, 233, 305, 440, 555, 655, 759, 806], 
                    "room0_seq2": [5, 80, 187, 296, 392, 497, 548, 658, 723], 
                    "room1_seq1": [17, 39, 125, 251, 349, 449, 542, 656, 840], 
                    "room2_seq1": [61, 178, 270, 323, 400, 485, 526, 601, 758], 
                }


                # 3 view, cannot cover the whole scene, can only cover a partial
                TRAIN_IDX_3v = {
                    "office2_seq2": [244, 291, 436], 
                    "office3_seq1": [22, 98, 315], 
                    "office4_seq2": [233, 305, 440], 
                    "room0_seq2": [392, 497, 658], 
                    "room1_seq1": [17, 39, 125], 
                    "room2_seq1": [323, 485, 526], 
                }

                if n_views == 6: 
                    TRAIN_IDX = TRAIN_IDX_FIXED
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX_FIXED
                    print(f"=> Replica 6 views. train_idx: {TRAIN_IDX}. ")
                
                # replica 3/9 views settings. corresponding to Table A1
                elif n_views == 9: 
                    TRAIN_IDX = TRAIN_IDX_9v
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX_FIXED
                    # testing views for 6/9 views are identical
                    print(f"=> Replica 9 views. train_idx: {TRAIN_IDX}. ")
                elif n_views == 3: 
                    TRAIN_IDX = TRAIN_IDX_3v
                    TRAIN_IDX_FOR_TEST = TRAIN_IDX
                    # for 3 view, only cover partial scene
                    print(f"=> Replica 3 views. train_idx: {TRAIN_IDX}. ")
            else: 

                # demo setting, not used commonly. only for better visualization on the project page. 
                TRAIN_IDX = {
                    "office2_seq2": [244, 291, 436, 574, 760, 831], 
                    "office3_seq1": [22, 98, 187, 315, 504, 581], 
                    "room0_seq2": [80, 187, 392, 497, 658, 833], 
                    "office4_seq1": [0, 242, 370, 401, 554, 822]
                }
            """
            
            TRAIN_IDX = {
                    "office2_seq2": [244, 291, 436, 607, 760, 831], 
                    "office3_seq1": [22, 98, 315, 504, 581, 731], 
                    "office4_seq2": [233, 305, 440, 555, 759, 806], 
                    "room0_seq2": [5, 80, 187, 392, 497, 658], 
                    "room1_seq1": [17, 39, 125, 349, 449, 840], 
                    "room2_seq1": [61, 178, 323, 485, 526, 758], 
                }
            
            TRAIN_IDX_FOR_TEST = TRAIN_IDX
            
            split = path.split("/")
            scene, seq = split[-2], split[-1]
            scene, scene_id = scene.split("_")[0], scene.split("_")[1]
            seq_id = seq.split("_")[1]
            train_idx = TRAIN_IDX[scene+scene_id+"_seq"+seq_id]
            train_idx_for_test_idx_generate = TRAIN_IDX_FOR_TEST[scene+scene_id+"_seq"+seq_id]
            scene = scene+scene_id+"_seq"+seq_id

            test_idx = []
            for idx in train_idx_for_test_idx_generate: 
                range_idx_left = [i for i in range(max(0, idx-50), idx)]
                range_idx_right = [i for i in range(idx+1, min(idx+50, len(cam_infos)))]
                range_idx = range_idx_left + range_idx_right
                range_idx = range_idx[::10] # NOTE
                test_idx.extend(range_idx)
            test_idx = list(set(test_idx))
            test_idx.sort()
            
            #train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            #test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
            #就是這裡在搞
            train_cam_infos = [c for c in cam_infos if extract_number(c.image_name) in train_idx]
            test_cam_infos  = [c for c in cam_infos if extract_number(c.image_name) in test_idx]

            breakpoint()
            
            
            if replica_use_project_cam: 
                gap = 6
                project_cam_infos = [cam_infos[idx] for idx in range(len(cam_infos))[::gap]]
                # NOTE: we only use the project cam to train baseline
            else: 
                project_cam_infos = None
        
        elif dataset == "Scannetpp": 
            if not demo_setting: 
                TRAIN_ID = {
                    "8a20d62ac0": [9, 85, 134, 172, 329, 380], 
                    "94ee15e8ba": [3057, 3107, 3177, 3184, 3274, 3302], 
                    "a29cccc784": [848, 865, 928, 947, 1006, 1040], 
                    "7831862f02": [3872, 3905, 3954, 3960, 3999, 4051], 
                }
            else: 
                TRAIN_ID = {
                    "94ee15e8ba": [3060, 3165, 3178, 3265, 3348, 3410], 
                    "7831862f02": [3890, 3926, 3941, 3996, 4050, 4172], 
                }
            TEST_INDICES_GAP = 6

            train_id = TRAIN_ID[scene_id]
            train_id.sort()
            
            frames_suffix = []
            for rgb_path in rgb_mapping: 
                suffix = rgb_path.split("/")[-1]
                suffix = suffix.split(".")[0]
                frames_suffix.append(int(suffix[3:8])) # '03870'

            # NOTE: it is different with train_id
            train_indices = [frames_suffix.index(ele) for ele in train_id]

            extend_range = 10
            start_frame_idx = max(train_indices[0]-extend_range, 0)
            end_frame_idx = min(train_indices[-1]+extend_range+1, len(cam_infos)) # not included

            test_indices = list(range(start_frame_idx, end_frame_idx))[::TEST_INDICES_GAP]
            for idx in train_indices: 
                if idx in test_indices: 
                    test_indices.remove(idx)
            
            train_idx = train_indices
            test_idx = test_indices

            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]

            project_cam_infos = None
        
        elif dataset == "re10k": 
            re10k_json_file = os.path.join(path, "train_test_split_{}.json".format(n_views))
            with open(re10k_json_file, "r") as f: 
                train_test_splits = json.load(f)

            train_idx = train_test_splits["train_ids"]
            test_idx = train_test_splits["test_ids"]
            print("=> re10k indices. train: {}. test: {}. ".format(train_idx, test_idx))
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
            
            project_cam_infos = None
        # ============================================================
        # 3) ⭐⭐ 通用版：不認得的 dataset 名稱，就照你說的資料夾規則來 ⭐⭐
        # ============================================================
        else:
            print(f"[info] unknown dataset '{dataset}', use generic COLMAP layout.")
            # 讀 train (sparse/0)
            try:
                train_img_file = os.path.join(path, "sparse/0", "images.bin")
                train_cam_file = os.path.join(path, "sparse/0", "cameras.bin")
                train_extr = read_extrinsics_binary(train_img_file)
                train_intr = read_intrinsics_binary(train_cam_file)
            except:
                train_img_file = os.path.join(path, "sparse/0", "images.txt")
                train_cam_file = os.path.join(path, "sparse/0", "cameras.txt")
                train_extr = read_extrinsics_text(train_img_file)
                train_intr = read_intrinsics_text(train_cam_file)

            # 讀 test (sparse/test) — 沒有就空 dict
            try:
                test_img_file = os.path.join(path, "sparse/test", "images.bin")
                test_cam_file = os.path.join(path, "sparse/test", "cameras.bin")
                test_extr = read_extrinsics_binary(test_img_file)
                test_intr = read_intrinsics_binary(test_cam_file)
            except:
                try:
                    test_img_file = os.path.join(path, "sparse/test", "images.txt")
                    test_cam_file = os.path.join(path, "sparse/test", "cameras.txt")
                    test_extr = read_extrinsics_text(test_img_file)
                    test_intr = read_intrinsics_text(test_cam_file)
                except:
                    test_extr, test_intr = {}, {}

            # 通用版就不搞 replica 的固定 index，直接把 images 目錄下面的檔名對過去
            train_dir = os.path.join(path, "images")
            test_dir  = os.path.join(path, "images_test")

            train_imgs = sorted(
                [f for f in glob.glob(os.path.join(train_dir, "*")) if f.lower().endswith(("jpg","jpeg","png"))]
            )
            test_imgs = sorted(
                [f for f in glob.glob(os.path.join(test_dir, "*")) if f.lower().endswith(("jpg","jpeg","png"))]
            )

            # 把 extrinsics 變成 {filename: ...}
            train_extr = {train_extr[k].name: train_extr[k] for k in train_extr} if len(train_extr) else {}
            test_extr  = {test_extr[k].name:  test_extr[k]  for k in test_extr}  if len(test_extr)  else {}

            # 真正組 CameraInfo
            train_cam_infos = []
            for idx, img_path in enumerate(train_imgs):
                fname = os.path.basename(img_path)
                if fname not in train_extr:
                    # 沒對到 colmap 的，就先跳過
                    continue
                extr = train_extr[fname]
                intr = train_intr[extr.camera_id]
                height = intr.height
                width = intr.width
                R = np.transpose(qvec2rotmat(extr.qvec))
                T = np.array(extr.tvec)

                if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                    fx = intr.params[0]
                    FovY = focal2fov(fx, height)
                    FovX = focal2fov(fx, width)
                elif intr.model == "PINHOLE":
                    fx, fy = intr.params[0], intr.params[1]
                    FovY = focal2fov(fy, height)
                    FovX = focal2fov(fx, width)
                else:
                    # 其他 model 就先簡單塞一個
                    fx = intr.params[0]
                    FovY = focal2fov(fx, height)
                    FovX = focal2fov(fx, width)

                img = Image.open(img_path).convert("RGB")
                cam = CameraInfo(
                    fid=idx,
                    uid=intr.id,
                    R=R, T=T,
                    FovY=FovY, FovX=FovX,
                    image=img,
                    image_path=img_path,
                    image_name=os.path.splitext(fname)[0],
                    width=width,
                    height=height,
                    mask=None,
                    bounds=np.array([1., 10.]),
                    projected_image=None,
                )
                train_cam_infos.append(cam)

            test_cam_infos = []
            for idx, img_path in enumerate(test_imgs):
                fname = os.path.basename(img_path)
                if fname not in test_extr:
                    continue
                extr = test_extr[fname]
                intr = test_intr[extr.camera_id]
                height = intr.height
                width = intr.width
                R = np.transpose(qvec2rotmat(extr.qvec))
                T = np.array(extr.tvec)

                if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                    fx = intr.params[0]
                    FovY = focal2fov(fx, height)
                    FovX = focal2fov(fx, width)
                elif intr.model == "PINHOLE":
                    fx, fy = intr.params[0], intr.params[1]
                    FovY = focal2fov(fy, height)
                    FovX = focal2fov(fx, width)
                else:
                    fx = intr.params[0]
                    FovY = focal2fov(fx, height)
                    FovX = focal2fov(fx, width)

                img = Image.open(img_path).convert("RGB")
                cam = CameraInfo(
                    fid=idx,
                    uid=intr.id,
                    R=R, T=T,
                    FovY=FovY, FovX=FovX,
                    image=img,
                    image_path=img_path,
                    image_name=os.path.splitext(fname)[0],
                    width=width,
                    height=height,
                    mask=None,
                    bounds=np.array([1., 10.]),
                    projected_image=None,
                )
                test_cam_infos.append(cam)

            project_cameras = None

    else:
        # not eval → 全部當 train
        train_cam_infos = cam_infos
        test_cam_infos = []
        project_cameras = None
    
    if get_dust3r_pcd: 
        return train_cam_infos
    
    if ply_path is not None: 
        pcd = fetchPly(ply_path)
    else: 
        pass

    for idx in range(len(train_cam_infos)):
        train_cam_infos[idx] = train_cam_infos[idx]._replace(fid=idx)
    #breakpoint()
    
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_indices=train_idx, 
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           all_cameras=cam_infos, 
                           project_cameras=project_cam_infos, 
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            mask = norm_data[:, :, 3:4]
            if skip == 1:
                depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                                      image_path.split('/')[-4]+'/'+image_name+'_depth.npy')
            else:
                depth_image = None

            arr = cv2.resize(arr, (400, 400))
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            depth_image = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask = None if mask is None else cv2.resize(mask, (400, 400))


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                        image_name=image_name, width=image.size[0], height=image.size[1],
                                        depth_image=depth_image, mask=mask))
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    pseudo_cam_infos = train_cam_infos #train_cam_infos
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 30000
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
