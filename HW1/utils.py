# utils.py
from pathlib import Path
import re, random, math
import cv2
import numpy as np
from typing import Union

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def set_seed(seed: int = 42):
    import torch, numpy as np, random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_image_by_id(image_id, img_dir):
    """
    支援 image_id 為純編號(如 '1') 或完整檔名(如 '1.jpg')
    """
    img_dir = Path(img_dir)
    s = str(image_id).strip()

    # 如果本來就帶副檔名，先直接檢查
    p = img_dir / s
    if p.exists() and p.suffix.lower() in IMG_EXTS:
        return p

    # 用 stem（拿掉副檔名）逐一嘗試
    stem = Path(s).stem
    for ext in IMG_EXTS:
        q = img_dir / f"{stem}{ext}"
        if q.exists():
            return q

    # 寬鬆一點的模糊匹配
    cands = [q for q in img_dir.iterdir()
             if q.suffix.lower() in IMG_EXTS and (q.stem == stem or stem in q.stem)]
    if cands:
        return sorted(cands, key=lambda x: x.name)[0]
    return None

def read_image_hw(img_path: Union[str, Path]):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return w, h

def clamp_box(x, y, w, h, W, H):
    """把 tlwh 夾回影像邊界；w,h <0 視為 0。"""
    x = max(0.0, min(float(x), W))
    y = max(0.0, min(float(y), H))
    w = max(0.0, float(w))
    h = max(0.0, float(h))
    # 如果超界，縮回來
    w = max(0.0, min(w, W - x))
    h = max(0.0, min(h, H - y))
    return x, y, w, h

def tlwh_to_yolo(x, y, w, h, W, H):
    """top-left width-height -> YOLO normalized cx,cy,w,h"""
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    return cx, cy, w / W, h / H

def xyxy_to_tlwh(x1, y1, x2, y2):
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

def format_pred_line(conf, x, y, w, h, cls_id=0):
    """輸出成：score x y w h class（score 6 位小數、座標兩位小數）"""
    return f"{conf:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {int(cls_id)}"

def parse_boxes_tokens(tokens):
    """
    支援：
      4欄重複:  x y w h                 -> 會自動補 cls=0
      5欄重複:  x y w h cls
      6欄重複:  score x y w h cls       -> 會忽略 score
    回傳 list[(x,y,w,h,cls)]
    """
    nums = [float(t) for t in tokens]  # 這裡 tokens 已經被 dataload 用正則切好
    n = len(nums)
    boxes = []
    if n == 0:
        return boxes

    if n % 6 == 0:
        g = 6
        for i in range(0, n, g):
            _s, x, y, w, h, c = nums[i:i+g]
            boxes.append((x, y, w, h, int(c)))
        return boxes

    if n % 5 == 0:
        g = 5
        for i in range(0, n, g):
            x, y, w, h, c = nums[i:i+g]
            boxes.append((x, y, w, h, int(c)))
        return boxes

    if n % 4 == 0:
        g = 4
        for i in range(0, n, g):
            x, y, w, h = nums[i:i+g]
            boxes.append((x, y, w, h, 0))  # 沒給類別就當作 pig=0
        return boxes

    raise ValueError(f"Token count {n} not divisible by 4/5/6")


def natural_int(s: str):
    try:
        return int(re.sub(r"[^0-9]", "", s))  # 抓數字部分做排序
    except:
        return None
