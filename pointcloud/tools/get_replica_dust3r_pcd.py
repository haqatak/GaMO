#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

# === 這三行是重點：把這支檔案的上層 (repo 根) 加進 sys.path ===
FILE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../getmask/tools
REPO_ROOT = os.path.dirname(FILE_DIR)                          # .../getmask
sys.path.append(REPO_ROOT)                                     # 可以 import scene, utils, tools, ...

# 如果你的 scene 在再上一層 (看你的結構是這樣) ─ e.g. /project2/.../getmask/scene
# 這裡順手也加
sys.path.append(os.path.join(REPO_ROOT, "scene"))
sys.path.append(os.path.join(REPO_ROOT, "utils"))
sys.path.append(os.path.join(REPO_ROOT, "tools"))
# === 重點結束 ===


from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import fov2focal
from tools.dust3r_to_colmap import convert_dust3r_to_colmap


def main():
    parser = argparse.ArgumentParser(
        description="Convert Replica/whatever scenes to DUSt3R-based COLMAP pcd."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="資料的根路徑，例如 /project2/.../Replica_3_master",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        required=True,
        help="要處理的 scene 名稱，例如 office_2 room_0",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Replica_3",
        help="dataset 名稱，給 readColmapSceneInfo 用，預設 Replica_3。",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="影像資料夾名稱，預設 images",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=3,
        help="訓練要挑的 view 數（原本你寫 3 或 6 那個），預設 3",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=1.0,
        help="dust3r_min_conf_thr，預設 1.0",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="是否使用 demo_setting 那組 dust3r 的輸出路徑",
    )
    parser.add_argument(
        "--dust3r_model_path",
        type=str,
        default="/project2/yichuanh/FOV-Outpainter/getmask/third_party/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        #default="/project2/yichuanh/FOV-Outpainter/Baseline/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        help="DUSt3R 權重路徑",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./dust3r_results",
        help="dust3r 結果要存在哪裡（外面那一層）",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="傳給 convert_dust3r_to_colmap 的 scale_factor，預設 1.0",
    )

    args = parser.parse_args()

    print("ROOT     :", args.root)
    print("SCENES   :", " ".join(args.scenes))
    print("DATASET  :", args.dataset)
    print("N_VIEWS  :", args.n_views)
    print("MIN_CONF :", args.min_conf)
    print("DEMO     :", args.demo)
    print("MODEL    :", args.dust3r_model_path)
    print("OUT_ROOT :", args.out_root)
    print("==============================")

    for scene_name in args.scenes:
        # 1) 真正的資料路徑
        scene_path = os.path.join(args.root, scene_name)
        print(f"🚀 run scene: {scene_name}")
        if not os.path.exists(scene_path):
            print(f"[warn] scene path not found: {scene_path}, skip.")
            continue
        
        #breakpoint()

        scene_key_for_save = scene_name  # e.g. "office_2"

        # 3) 決定 dust3r 影像要存哪
        if args.demo:
            img_dir_for_dust3r = os.path.join(
                args.out_root,
                f"{args.dataset}",
                scene_key_for_save,
            )
        else:
            img_dir_for_dust3r = os.path.join(
                args.out_root,
               f"{args.dataset}",
                scene_key_for_save,
            )
        #breakpoint()
        os.makedirs(img_dir_for_dust3r, exist_ok=True)

        # 4) 先用你改過的 dataset_readers 去抓 train_cam_infos
        train_cam_infos = readColmapSceneInfo(
            scene_path,
            args.images,
            args.dataset,
            True,  # eval
            args.n_views,
            args.min_conf,
            args.demo,
            get_dust3r_pcd=True,
        )

        # 5) 把 camera 轉成你要的形式
        train_img_paths = [c.image_path for c in train_cam_infos]
        known_c2w, known_focal = [], []
        for c in train_cam_infos:
            Rt = np.zeros((4, 4), dtype=np.float32)
            Rt[:3, :3] = c.R.transpose()
            Rt[:3, 3] = c.T
            Rt[3, 3] = 1.0

            C2W = np.linalg.inv(Rt)
            known_c2w.append(C2W)

            # note: 這裡跟你原本一樣的做法，把 focal normalized 到 512
            fx = fov2focal(c.FovX, c.width)
            scale = max(c.width, c.height) / 512.0
            known_focal.append(fx / scale)

        # 6) 真正呼叫 convert_dust3r_to_colmap
        convert_dust3r_to_colmap(
            image_files=train_img_paths,
            save_dir=img_dir_for_dust3r,
            min_conf_thr=args.min_conf,
            model_path=args.dust3r_model_path,
            known_c2w=known_c2w,
            known_focal=known_focal,
            no_mask_pc=False,
            scale_factor=args.scale_factor,
        )
        
        #breakpoint()
        
        print(f"[done] scene {scene_name} → saved to {img_dir_for_dust3r}")


if __name__ == "__main__":
    main()
