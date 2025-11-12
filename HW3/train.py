# train.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim

from dataload import get_mnist_folder_loader
from models import UNet28
from diffusion import GaussianDiffusion

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--save", type=str, default="ddpm_mnist.pt")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_mnist_folder_loader(root="./mnist",
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True, drop_last=True)
    model = UNet28().to(device)
    ddpm = GaussianDiffusion(model, timesteps=args.timesteps, device=device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, ddpm.T, (x.size(0),), device=device).long()
            loss = ddpm.p_losses(x, t)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1
            pbar.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "timesteps": ddpm.T
    }, args.save)
    print(f"[OK] saved checkpoint -> {args.save}")

if __name__ == "__main__":
    main()
