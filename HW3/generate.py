# sample.py
import os
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from models import UNet28
from diffusion import GaussianDiffusion

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ddpm_mnist.pt")
    p.add_argument("--out_dir", type=str, default="gen_imgs")
    p.add_argument("--num", type=int, default=10000)          # 產生張數
    p.add_argument("--batch_size", type=int, default=256)     # 抽樣 batch
    p.add_argument("--zip_name", type=str, default="images_<student-id>.zip")
    p.add_argument("--make_grid", action="store_true", help="輸出 8x8 diffusion grid")
    p.add_argument("--timesteps", type=int, default=None, help="覆蓋 ckpt timesteps（通常不用）")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load model -----
    model = UNet28().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    T = args.timesteps if args.timesteps is not None else ckpt.get("timesteps", 1000)
    ddpm = GaussianDiffusion(model, timesteps=T, device=device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- (A) optional diffusion grid -----
    if args.make_grid:
        snaps = ddpm.sample_with_checkpoints(n=8, checkpoints=7)
        grid = GaussianDiffusion.make_report_grid(snaps)  # [3,H,W] in [0,1]
        save_image(grid, out_dir / "diffusion_grid.png")
        print(f"[OK] saved diffusion grid -> {out_dir/'diffusion_grid.png'}")

    # ----- (B) generate N images, 00001.png ... N.png -----
    remain = args.num
    cur_id = 1
    pbar = tqdm(total=remain, desc="Generating")
    while remain > 0:
        b = min(args.batch_size, remain)
        imgs = ddpm.sample(b, batch_size=b)                 # [-1,1]
        imgs = (imgs.clamp(-1,1) + 1.0) * 0.5              # -> [0,1]
        for i in range(b):
            fname = f"{cur_id:05d}.png"
            save_image(imgs[i], out_dir / fname)
            cur_id += 1
        remain -= b
        pbar.update(b)
    pbar.close()
    print(f"[OK] generated {args.num} images to {out_dir.resolve()}")

    # ----- (C) zip WITHOUT subdirectory -----
    zip_path = Path(args.zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("*.png")):
            zf.write(p, arcname=p.name)   # arcname 僅檔名 -> 無子資料夾
    print(f"[OK] zipped -> {zip_path.resolve()} (flat, no subdirectories)")

if __name__ == "__main__":
    main()
