# sample.py — same CLI, faster sampling & IO (AMP + threaded saves + no-compress ZIP)
import os
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm
import contextlib
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import torch
from torchvision.transforms.functional import to_pil_image

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


def _save_png(img_tensor_cpu_01, path):
    """
    單張存檔：img_tensor_cpu_01 是 CPU、float、[0,1] 的 [3,28,28]
    使用 PIL，設定 compress_level=1（極快）。
    """
    im = to_pil_image(img_tensor_cpu_01)  # tensor->[0,1] to PIL RGB 8-bit
    im.save(path, compress_level=1)


@torch.inference_mode()  # 比 no_grad 更省開銷
def main():
    args = parse()

    # -------------------- GPU & 內核優化 --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] cuda={torch.cuda.is_available()} using={device}")

    torch.backends.cudnn.benchmark = True  # 小圖卷積通常更快

    # AMP 自動混合精度（僅在有 CUDA 時啟用）
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    # ----- load model -----
    model = UNet28().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])   # 先載權重
    model.eval()                           # 先切 eval

    # 再 compile（避免 _orig_mod.* 鍵名不對）
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
            print("[Info] torch.compile enabled after load.")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")

    # compile 完再交給 diffusion 包裝
    T = args.timesteps if args.timesteps is not None else ckpt.get("timesteps", 1000)
    ddpm = GaussianDiffusion(model, timesteps=T, device=device)


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- (A) optional diffusion grid -----
    if args.make_grid:
        with amp_ctx:
            snaps = ddpm.sample_with_checkpoints(n=8, checkpoints=7)
        from torchvision.utils import save_image  # 僅這裡用一下
        grid = GaussianDiffusion.make_report_grid(snaps)  # [3,H,W] in [0,1]
        save_image(grid, "diffusion_grid.png")
        print(f"[OK] saved diffusion grid -> {'diffusion_grid.png'}")

    # -------------------- (B) generate N images --------------------
    remain = args.num
    cur_id = 1
    pbar = tqdm(total=remain, desc="Generating")

    # 檔案寫入用 thread pool（PNG 存檔會釋放 GIL，thread 很有效）
    max_workers = min(8, (os.cpu_count() or 8))  # 你可改大一些（12/16），視 CPU 而定
    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending = deque()  # 控制 flight 中的任務數，避免佔太多 RAM

    try:
        while remain > 0:
            b = min(args.batch_size, remain)

            # ---- GPU 抽樣 (與 CPU 存檔重疊) ----
            with amp_ctx:
                imgs = ddpm.sample(b, batch_size=b)  # [-1,1]，回傳在 CPU（diffusion 寫的是 .cpu()）
            imgs = (imgs.clamp(-1, 1) + 1.0) * 0.5   # -> [0,1] CPU float tensor

            # ---- 將存檔工作丟給 thread pool，馬上進下一個 batch ----
            # 為避免任務爆量，限制 flight 中的工作數量（4x worker）
            limit = max_workers * 4
            for i in range(b):
                fname = f"{cur_id:05d}.png"
                fut = executor.submit(_save_png, imgs[i], out_dir / fname)
                pending.append(fut)
                cur_id += 1

                if len(pending) > limit:
                    # 等最早的一批完成，釋放記憶體
                    pending.popleft().result()

            remain -= b
            pbar.update(b)

        # 等待所有存檔完成
        while pending:
            pending.popleft().result()

    finally:
        executor.shutdown(wait=True)
        pbar.close()

    print(f"[OK] generated {args.num} images to {out_dir.resolve()}")

    # -------------------- (C) zip (不壓縮，超快) --------------------
    zip_path = Path(args.zip_name)
    # 若你一定要壓縮，改回 ZIP_DEFLATED，但會慢很多
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out_dir.glob("*.png")):
            zf.write(p, arcname=p.name)  # 扁平結構
    print(f"[OK] zipped -> {zip_path.resolve()} (flat, no subdirectories)")


if __name__ == "__main__":
    main()
