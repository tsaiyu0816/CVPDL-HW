# diffusion.py
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from dataload import denorm_to_uint8

class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size=28, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        super().__init__()
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        # register buffers
        for name, buf in [
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_cumprod", alphas_cumprod),
            ("alphas_cumprod_prev", alphas_cumprod_prev),
            ("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)),
            ("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)),
            ("sqrt_recip_alphas", torch.sqrt(1.0 / alphas)),
            ("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20)),
        ]:
            self.register_buffer(name, buf)

        self.to(self.device)

    # -------- forward (q) --------
    def q_sample(self, x0, t, noise=None):
        """
        x0: (B,3,28,28) in [-1,1]
        t:  (B,) long
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_om * noise

    # -------- training loss --------
    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = self.model(xt, t)
        return torch.mean((noise - pred_noise) ** 2)

    # -------- reverse step --------
    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        one step: t -> t-1
        """
        b = xt.shape[0]
        eps_theta = self.model(xt, t)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        beta_t  = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1,1,1,1)
        sqrt_om_cum_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)

        # Eq. 11 in DDPM paper
        mean = sqrt_recip_alpha_t * (xt - beta_t / sqrt_om_cum_t * eps_theta)

        if (t == 0).all():
            return mean
        else:
            noise = torch.randn_like(xt)
            var = self.posterior_variance[t].view(-1,1,1,1)
            return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n, batch_size=256):
        """
        Sample n images, return float tensor in [-1,1]
        """
        imgs = []
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            x = torch.randn(b, 3, 28, 28, device=self.device)
            for t in reversed(range(self.T)):
                t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t_batch)
            imgs.append(x.cpu())
            remaining -= b
        return torch.cat(imgs, dim=0)

    @torch.no_grad()
    def sample_with_checkpoints(self, n=8, checkpoints=7):
        """
        For the report grid:
        - Generate n samples in parallel.
        - Divide T into `checkpoints` equal parts.
        - Return a list of tensors [snap0, snap1, ..., snap_checkpoints] each (n,3,28,28)
          from early noisy states down to final results.
        """
        x = torch.randn(n, 3, 28, 28, device=self.device)
        # indices to store (descending order of t)
        # e.g., T=1000, checkpoints=7 -> [999, 857, 714, 571, 428, 285, 142, 0]
        step = max(1, self.T // 7)
        target_ts = list(reversed([i for i in range(0, self.T, step)][:7])) + [0]
        snaps = []

        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
            if t in target_ts:
                snaps.append(x.detach().cpu())

        # snaps are in descending t; ensure length 8
        snaps = snaps + [x.detach().cpu()] if len(snaps) < 8 else snaps[:8]
        return snaps  # length 8 list, each (n,3,28,28)

    @staticmethod
    def make_report_grid(snaps):
        """
        snaps: list length 8, each is (N,3,28,28) with N==8.
        Arrange to 8x8 grid (rows=time, cols=samples).
        """
        # we want rows=time flowing downward -> stack each time snapshot as a row of 8
        rows = []
        for s in snaps:  # each s: (8,3,28,28)
            # convert to [0,1] for grid
            row = (s.clamp(-1,1) + 1) * 0.5
            row = make_grid(row, nrow=row.shape[0], padding=2)  # 8 across
            rows.append(row)
        grid = torch.cat(rows, dim=1)  # stack vertical
        return grid
