#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
import sys

MODELS_WITH_FX_FY = {
    "PINHOLE",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "RADIAL",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
    "FOV",
}

def copytree_compat(src: Path, dst: Path, overwrite: bool = False):
    """相容 Python 3.7 的 copytree：若存在就合併覆蓋。"""
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = src / item.name
        d = dst / item.name
        if s.is_dir():
            copytree_compat(s, d, overwrite=overwrite)
        else:
            if not d.exists() or overwrite:
                shutil.copy2(s, d)

def try_float(x: str):
    try:
        return float(x)
    except:
        return None

def scale_fx_fy_in_cameras(cameras_path: Path, scale: float = 0.6):
    if not cameras_path.exists():
        raise FileNotFoundError(f"{cameras_path} not found")

    lines = cameras_path.read_text(encoding="utf-8").splitlines()
    out_lines, changed = [], 0

    for line in lines:
        if not line.strip() or line.startswith("#"):
            out_lines.append(line)
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            out_lines.append(line)
            continue
        model = parts[1].upper()
        if model in MODELS_WITH_FX_FY:
            fx, fy = try_float(parts[4]), try_float(parts[5])
            if fx is not None and fy is not None:
                fx *= scale
                fy *= scale
                parts[4] = f"{fx:.12f}"
                parts[5] = f"{fy:.12f}"
                changed += 1
        out_lines.append(" ".join(parts))

    cameras_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed

def process_scene(base_root: Path, scene: str, scale: float, overwrite: bool):
    scene_root = base_root / scene
    images_src = scene_root / "images"
    images_dst = scene_root / "images_test"
    sparse0_src = scene_root / "sparse" / "0"
    sparset_dst = scene_root / "sparse" / "test"
    cameras_txt = sparset_dst / "cameras.txt"

    print(f"\n=== Processing: {scene_root} ===")

    # 1. copy images
    print(" [*] Copy images -> images_test")
    copytree_compat(images_src, images_dst, overwrite=overwrite)

    # 2. copy sparse
    print(" [*] Copy sparse/0 -> sparse/test")
    copytree_compat(sparse0_src, sparset_dst, overwrite=overwrite)

    # 3. modify fx, fy
    print(f" [*] Scale fx, fy by {scale} in {cameras_txt}")
    changed = scale_fx_fy_in_cameras(cameras_txt, scale)
    print(f" [✓] Modified {changed} lines with fx/fy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_root = Path("data") / args.mode / args.root
    if not base_root.exists():
        print(f"❌ Base root not found: {base_root}")
        sys.exit(1)

    print(f"Base root: {base_root}")
    print(f"Scenes   : {', '.join(args.scenes)}")
    print(f"Scale    : {args.scale}")
    print(f"Overwrite: {args.overwrite}")

    for scene in args.scenes:
        try:
            process_scene(base_root, scene, args.scale, args.overwrite)
        except Exception as e:
            print(f" [!] Error processing scene '{scene}': {e}", file=sys.stderr)

    print("\nAll done.")

if __name__ == "__main__":
    main()
