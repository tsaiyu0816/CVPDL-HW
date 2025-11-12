# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- time embedding -----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) int/float timesteps
        return: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb   = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.emb(t).unsqueeze(-1).unsqueeze(-2)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.block1 = ResBlock(ch, ch, t_dim)
        self.block2 = ResBlock(ch, ch, t_dim)
        self.down   = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.up     = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.block1 = ResBlock(ch*2, ch, t_dim)
        self.block2 = ResBlock(ch, ch, t_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        return x

class UNet28(nn.Module):
    """
    Tiny UNet for 28x28 RGB, noise-prediction (epsilon) model
    """
    def __init__(self, in_ch=3, base=64, t_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*4),
            nn.SiLU(),
            nn.Linear(t_dim*4, t_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = Down(base, t_dim)        # 28 -> 14
        self.down2 = Down(base, t_dim)        # 14 -> 7

        self.mid1  = ResBlock(base, base, t_dim)
        self.mid2  = ResBlock(base, base, t_dim)

        self.up2   = Up(base, t_dim)          # 7 -> 14
        self.up1   = Up(base, t_dim)          # 14 -> 28

        self.out   = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_emb(t)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t)
        x, s2 = self.down2(x, t)

        x = self.mid1(x, t)
        x = self.mid2(x, t)

        x = self.up2(x, s2, t)
        x = self.up1(x, s1, t)

        return self.out(x)
