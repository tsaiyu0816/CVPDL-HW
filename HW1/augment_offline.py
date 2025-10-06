# pip install albumentations opencv-python
import os, glob, cv2, random
from pathlib import Path
import albumentations as A

SRC_IMG = "data/train/images"
SRC_LB  = "data/train/labels"
DST_IMG = "data/train_aug/images"
DST_LB  = "data/train_aug/labels"
os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LB, exist_ok=True)

T = A.Compose([
    A.HorizontalFlip(p=0.8),  # 補左↔右
    A.OneOf([
        A.RandomGamma(gamma_limit=(40, 160), p=0.7),            # 日↔夜
        A.RandomBrightnessContrast(brightness_limit=0.35,
                                   contrast_limit=0.35, p=0.7),
        A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8), p=0.3)
    ], p=1.0),
    A.OneOf([
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.5)
    ], p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.3),
        A.Defocus(radius=(2, 4), p=0.3),
        A.ImageCompression(quality_lower=35, quality_upper=85, p=0.4)
    ], p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),  # 輕微透視
    A.CoarseDropout(max_holes=4, max_height=0.08, max_width=0.08, p=0.3)
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))

def read_labels(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    out = []
    for l in lines:
        c, x, y, w, h = l.split()
        out.append([float(x), float(y), float(w), float(h), int(c)])
    return out

def write_labels(path, bboxes):
    with open(path, "w") as f:
        for x, y, w, h, c in bboxes:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

imgs = sorted(glob.glob(os.path.join(SRC_IMG, "*.*")))
for img_path in imgs:
    stem = Path(img_path).stem
    lb_path = os.path.join(SRC_LB, stem + ".txt")
    bbs = read_labels(lb_path)
    if not bbs:
        continue
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    boxes = [[bx, by, bw, bh] for bx,by,bw,bh,_ in bbs]
    labels = [c for *_, c in bbs]

    # 每張產 1~2 份
    for k in range(random.choice([1,2])):
        aug = T(image=img, bboxes=boxes, class_labels=labels)
        oimg, obox, olab = aug["image"], aug["bboxes"], aug["class_labels"]
        if len(obox)==0:
            continue
        out_stem = f"{stem}_aug{k}"
        cv2.imwrite(os.path.join(DST_IMG, out_stem + ".jpg"), oimg)
        write_labels(os.path.join(DST_LB, out_stem + ".txt"),
                     [[*b, c] for b, c in zip(obox, olab)])
print("Done.")
