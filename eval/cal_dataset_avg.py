import json
import os

from argparse import ArgumentParser


parser = ArgumentParser(description="Relighting script parameters")
parser.add_argument("--root", default="", type=str)
args = parser.parse_args()

root = args.root

ssim = 0.0
psnr = 0.0
lpips = 0.0

dirpaths = [dirpath for dirpath in os.listdir(root) if os.path.isdir(f"{root}/{dirpath}")]
total = len(dirpaths)
print(f"Root: {args.root}, Number of scenes: {total}")

for dirpath in dirpaths:
    with open(f"{root}/{dirpath}/relight/relit_avg.json", "r") as fr:
        json_file = json.load(fr)
    ssim += json_file["SSIM"]
    psnr += json_file["PSNR"]
    lpips += json_file["LPIPS"]

with open(f"{root}/relit_avg.json", "w") as fw:
    json.dump({"PSNR": psnr / total, "SSIM": ssim / total, "LPIPS": lpips / total}, fw)
