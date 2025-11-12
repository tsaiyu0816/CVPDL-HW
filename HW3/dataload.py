# data.py
import os, glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ALLOWED_EXTS = (".png")

def _tfm_28_rgb_to_minus1_1():
    return transforms.Compose([
        transforms.ToTensor(),                                # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])# -> [-1,1]
    ])

class ImageFolderNoLabel(Dataset):
    """讀取 ./mnist/*.png|jpg… 的 3×28×28 影像，不使用 label。"""
    def __init__(self, root="./mnist"):
        self.root = Path(root)
        files = []
        for ext in ALLOWED_EXTS:
            files += glob.glob(str(self.root / f"*{ext}"))
        self.files = sorted(files)
        assert len(self.files) > 0, f"No images found in {self.root}"
        self.tfm = _tfm_28_rgb_to_minus1_1()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")     # 保險：確保 3 通道
        # 確認大小；不是 28×28 就報錯（避免不小心 resize 破壞題目）
        if img.size != (28, 28):
            raise ValueError(f"Image {p} has size {img.size}, expected (28,28).")
        x = self.tfm(img)                      # tensor in [-1,1], shape [3,28,28]
        return x, 0                            # label 不用，回傳個占位 0

def get_mnist_folder_loader(root="./mnist", batch_size=128, num_workers=2,
                            shuffle=True, drop_last=True):
    ds = ImageFolderNoLabel(root=root)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)

def denorm_to_uint8(x):
    """把 [-1,1] tensor 轉 uint8 [0,255]（若你之後需要手動存圖會用到）。"""
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    return (x * 255.0).clamp(0, 255).to(torch.uint8)
