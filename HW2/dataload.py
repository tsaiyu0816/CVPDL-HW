#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataload.py
- 轉換 train/test 影像與標註為 YOLO 格式
- （可選）自動切出 val（固定 seed）
- ValSync：把與 val 同名的 train 影像/標註（以及 chips）移除
- 依「目前的 train」再生 chips（避免洩漏到 val）
- 產生 data.yaml，若 chips 存在則 train 指向 ["images/train","images/train_chips"]

建議呼叫（例）：
python dataload.py \
  --src data/hw2/CVPDL_hw2/CVPDL_hw2 \
  --dst data/hw2_yolo \
  --yaml data/hw2_fold0.yaml \
  --names "car,hov,person,motorcycle" \
  --val_ratio 0.10 --val_seed 2025 --skip_val_if_exists 1 \
  --make_chips --chip_size 1024 --chips_per_img 2 --chip_format jpg --jpg_quality 90 \
  --clear_old_chips 1 \
  --workers 12
"""
import argparse, shutil, yaml, random, re, os, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from typing import Optional, Tuple, List

# ---------- 共用 ----------
IMG_EXTS = (".jpg",".jpeg",".png",".JPG",".PNG",".JPEG")

def _write_data_yaml(yaml_path, root, names, include_chips):
    d = {
        "path": str(Path(root)),
        "train": ["images/train", "images/train_chips"] if include_chips else "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i:n for i,n in enumerate(names)},
        "nc": len(names),
    }
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f: yaml.safe_dump(d, f, sort_keys=False)
    print(f"[ok] wrote {yaml_path} | train uses: {d['train']}")

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _parse_label_line(line: str, W: int, H: int) -> Optional[Tuple[int,float,float,float,float]]:
    """
    接受：
      1) 'c cx cy w h'  (0~1 正規化，空白或逗號)
      2) 'c,x,y,w,h'    (像素 XYWH)
      3) 'c,xmin,ymin,xmax,ymax' (像素 XYXY)
    回傳 YOLO 正規化 (cid,cx,cy,w,h)；失敗回 None
    """
    s = line.strip()
    if not s: return None
    toks = re.split(r"[,\s]+", s)
    if len(toks) != 5:
        return None
    try:
        cid = int(float(toks[0]))
        a,b,c,d = map(float, toks[1:])
    except Exception:
        return None

    # 優先判斷正規化
    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1 and 0 <= d <= 1:
        return cid, float(a), float(b), float(c), float(d)

    # 再判斷 XYXY
    if d > b and c > a:
        xmin, ymin, xmax, ymax = a, b, c, d
        xmin = _clip(xmin, 0, W); xmax = _clip(xmax, 0, W)
        ymin = _clip(ymin, 0, H); ymax = _clip(ymax, 0, H)
        ww = max(0.0, xmax - xmin); hh = max(0.0, ymax - ymin)
        if ww <= 0 or hh <= 0: return None
        cx = (xmin + ww/2.0) / W
        cy = (ymin + hh/2.0) / H
        w  = ww / W
        h  = hh / H
        return cid, float(cx), float(cy), float(w), float(h)

    # 最後當成 XYWH
    x, y, wpx, hpx = a, b, c, d
    x = _clip(x, 0, W); y = _clip(y, 0, H)
    wpx = max(0.0, min(wpx, W)); hpx = max(0.0, min(hpx, H))
    if wpx <= 0 or hpx <= 0: return None
    cx = (x + wpx/2.0) / W
    cy = (y + hpx/2.0) / H
    w  = wpx / W
    h  = hpx / H
    return cid, float(cx), float(cy), float(w), float(h)

def _img_stem(p: str):
    b = os.path.basename(p)
    s = os.path.splitext(b)[0]
    for ext in IMG_EXTS:
        if b.endswith(ext): return b[:-len(ext)]
    return s

# ---------- 影像/標註轉換（並行） ----------
def _convert_one(train_src_dir: Path, out_img_dir: Path, out_lbl_dir: Path, p: Path) -> Optional[str]:
    """複製影像 + 解析標註成 YOLO 正規化；失敗回傳檔名方便 log。"""
    try:
        shutil.copy2(p, out_img_dir / p.name)
        lbl_src = train_src_dir / (p.stem + ".txt")
        if not lbl_src.exists():
            return None  # 沒標註就跳過（不寫 label）
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            return p.name
        H, W = im.shape[:2]
        y_lines=[]
        for ln in lbl_src.read_text().splitlines():
            parsed = _parse_label_line(ln, W, H)
            if parsed is None:
                continue
            cid, cx, cy, w, h = parsed
            cx = _clip(cx, 0, 1); cy = _clip(cy, 0, 1)
            w  = _clip(w,  0, 1); h  = _clip(h,  0, 1)
            if w<=0 or h<=0:
                continue
            y_lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if y_lines:
            (out_lbl_dir/(p.stem+".txt")).write_text("\n".join(y_lines), encoding="utf-8")
        return None
    except Exception:
        return p.name  # return failed image name

# ---------- chips 擴增（並行） ----------
def _yolo_to_abs(lines: List[str], W:int, H:int, min_area:int=4):
    anns=[]
    for ln in lines:
        ln=ln.strip()
        if not ln: continue
        try:
            cid, cx, cy, w, h = map(float, re.split(r"\s+", ln))
            bx = int((cx - w/2) * W)
            by = int((cy - h/2) * H)
            bw = int(w * W); bh = int(h * H)
            if bw*bh < min_area: continue
            anns.append((int(cid), bx,by,bw,bh))
        except Exception:
            continue
    return anns

def _save_chip(out_img_path: Path, chip: np.ndarray, fmt: str, jpg_q: int, png_c: int):
    if fmt == "jpg":
        cv2.imwrite(str(out_img_path.with_suffix(".jpg")), chip, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_q)])
    else:
        cv2.imwrite(str(out_img_path.with_suffix(".png")), chip, [cv2.IMWRITE_PNG_COMPRESSION, int(png_c)])

def _chips_from_one(lbl: Path, img_tr: Path, out_img: Path, out_lbl: Path,
                    chip_size:int, chips_per_img:int, jitter:float,
                    min_area:int, fmt:str, jpg_q:int, png_c:int, seed:int):
    """對單一標註檔產生 chips。"""
    if chips_per_img <= 0: return 0
    rng = random.Random(seed ^ hash(lbl.stem))
    stem = lbl.stem
    # 找對應影像
    img=None
    for ext in (".png",".jpg",".jpeg",".JPG",".PNG",".JPEG"):
        p=img_tr/f"{stem}{ext}"
        if p.exists(): img=str(p); break
    if img is None: return 0
    im=cv2.imread(img, cv2.IMREAD_COLOR)
    if im is None: return 0
    H,W = im.shape[:2]
    lines=[ln for ln in lbl.read_text().splitlines() if ln.strip()]
    if not lines: return 0
    anns=_yolo_to_abs(lines, W, H, min_area=min_area)
    if not anns: return 0

    made=0
    for k in range(chips_per_img):
        persons = [a for a in anns if a[0] == 2]  # class 2 = person
        if persons and rng.random() < 0.6:
            cid,x,y,ww,hh = rng.choice(persons)
        else:
            cid,x,y,ww,hh = rng.choice(anns)

        cx0 = x + ww//2; cy0 = y + hh//2
        jitter_px = int(chip_size * max(0.0, min(1.0, jitter)))
        cx0 = max(chip_size//2, min(W - chip_size//2, cx0 + rng.randint(-jitter_px,jitter_px)))
        cy0 = max(chip_size//2, min(H - chip_size//2, cy0 + rng.randint(-jitter_px,jitter_px)))

        x1 = max(0, cx0 - chip_size//2); y1 = max(0, cy0 - chip_size//2)
        x1 = min(x1, max(0, W - chip_size)); y1 = min(y1, max(0, H - chip_size))
        x2 = min(W, x1 + chip_size); y2 = min(H, y1 + chip_size)

        crop = im[y1:y2, x1:x2].copy()
        new_lines=[]
        chip_x2 = x1 + chip_size
        chip_y2 = y1 + chip_size
        for (cid2, bx, by, bw, bh) in anns:
            bx1, by1 = bx, by
            bx2, by2 = bx + bw, by + bh
            ix1 = max(bx1, x1)
            iy1 = max(by1, y1)
            ix2 = min(bx2, chip_x2)
            iy2 = min(by2, chip_y2)
            iw = ix2 - ix1
            ih = iy2 - iy1
            if iw <= 0 or ih <= 0 or iw * ih < min_area:
                continue
            nx1 = ix1 - x1
            ny1 = iy1 - y1
            xc  = (nx1 + iw/2.0) / chip_size
            yc  = (ny1 + ih/2.0) / chip_size
            ww2 = iw / chip_size
            hh2 = ih / chip_size
            if ww2 <= 0 or hh2 <= 0:
                continue
            xc  = max(0.0, min(1.0, xc))
            yc  = max(0.0, min(1.0, yc))
            ww2 = max(0.0, min(1.0, ww2))
            hh2 = max(0.0, min(1.0, hh2))
            new_lines.append(f"{cid2} {xc:.6f} {yc:.6f} {ww2:.6f} {hh2:.6f}")

        if not new_lines:
            continue

        out_name = f"{stem}__chip{k+1}"
        out_img_path = out_img / out_name
        _save_chip(out_img_path, crop, fmt, jpg_q, png_c)
        (out_lbl/f"{out_name}.txt").write_text("\n".join(new_lines), encoding="utf-8")
        made += 1
    return made

# ---------- 輔助：資料集維護 ----------
def _glob_images(d: str):
    lst=[]
    for ext in IMG_EXTS:
        lst += glob.glob(os.path.join(d, f"*{ext}"))
    return sorted(lst)

def _val_exists_and_nonempty(root: Path) -> bool:
    vi = root/"images/val"
    return vi.exists() and any(_glob_images(str(vi)))

def _split_val_if_needed(root: Path, ratio: float, seed: int, skip_if_exists: bool=True):
    if ratio <= 0:
        print("[ValSplit] ratio<=0, skip.")
        return 0
    if skip_if_exists and _val_exists_and_nonempty(root):
        print("[ValSplit] found existing val, skip.")
        return 0

    ti, tl = root/"images/train", root/"labels/train"
    vi, vl = root/"images/val",   root/"labels/val"
    vi.mkdir(parents=True, exist_ok=True); vl.mkdir(parents=True, exist_ok=True)

    imgs = _glob_images(str(ti))
    k = max(1, round(ratio * len(imgs)))
    random.seed(seed)
    val_set = set(random.sample(imgs, k))

    def stem(p):
        return _img_stem(p)

    moved=0
    for p in imgs:
        s = stem(p)
        lp = tl/(s + ".txt")
        if p in val_set:
            shutil.move(p, vi/os.path.basename(p)); moved+=1
            if lp.exists():
                shutil.move(str(lp), vl/os.path.basename(lp))
    print(f"[ValSplit] moved {moved} images to val.")
    return moved

def _valsync_remove_overlaps(root: Path):
    ti, tl = root/"images/train", root/"labels/train"
    vi, vl = root/"images/val",   root/"labels/val"
    tci,tcl = root/"images/train_chips", root/"labels/train_chips"

    def stems(d: Path):
        ss=set()
        for p in _glob_images(str(d)):
            ss.add(_img_stem(p))
        return ss

    if not (vi.exists() and any(_glob_images(str(vi)))):
        print("[ValSync] val empty, skip.")
        return (0,0)

    val_stems = stems(vi)

    # 刪 train / labels/train
    n_rm=0
    for s in list(val_stems):
        # image
        for ext in IMG_EXTS:
            p = ti/(s+ext)
            if p.exists():
                os.remove(p); n_rm+=1; break
        # label
        lp = tl/(s+".txt")
        if lp.exists(): os.remove(lp)

    # 刪 chips（__chipK）
    n_chip_rm=0
    for d in (tci, tcl):
        if d.exists():
            for s in val_stems:
                for p in d.glob(f"{s}__chip*.*"):
                    os.remove(p); n_chip_rm+=1
    print(f"[ValSync] removed {n_rm} train files and {n_chip_rm} chips that overlap with val")
    return (n_rm, n_chip_rm)

def _build_chips_after_val(root: Path, chip_size:int, chips_per_img:int, jitter:float,
                           min_area:int, fmt:str, jpg_q:int, png_c:int, seed:int,
                           clear_old: bool, workers: int):
    img_tr, lbl_tr = root/"images/train", root/"labels/train"
    img_ch, lbl_ch = root/"images/train_chips", root/"labels/train_chips"
    img_ch.mkdir(parents=True, exist_ok=True)
    lbl_ch.mkdir(parents=True, exist_ok=True)

    if clear_old:
        for d in (img_ch, lbl_ch):
            for p in d.glob("*"):
                try: os.remove(p)
                except IsADirectoryError: pass
        print("[Chips] cleared old chips.")

    lbl_files = sorted(lbl_tr.glob("*.txt"))
    print(f"[*] building chips with {workers} workers ... labels={len(lbl_files)} "
          f"(size={chip_size}, per_img={chips_per_img}, fmt={fmt})")
    total_made = 0
    if chips_per_img <= 0:
        print("[Chips] chips_per_img<=0, skip.")
        return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _chips_from_one, lbl, img_tr, img_ch, lbl_ch,
                chip_size, chips_per_img, jitter,
                min_area, fmt, jpg_q, png_c,
                seed + i
            )
            for i, lbl in enumerate(lbl_files)
        ]
        for i,f in enumerate(as_completed(futs), 1):
            total_made += f.result()
            if i % 200 == 0 or i == len(lbl_files):
                print(f"  - progress {i}/{len(lbl_files)} (chips={total_made})", flush=True)
    print(f"[ok] chips generated: {total_made}")
    return total_made

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../CVPDL_hw2/CVPDL_hw2")
    ap.add_argument("--dst", required=True, help="yolo root (will be created)")
    ap.add_argument("--yaml", required=True, help="where to write the final data.yaml (train+chips, val, test)")
    ap.add_argument("--names", default="car,hov,person,motorcycle")

    # 並行
    ap.add_argument("--workers", type=int, default=8)

    # val split / sync
    ap.add_argument("--val_ratio", type=float, default=0.10, help="0~1; 0=不切")
    ap.add_argument("--val_seed", type=int, default=2025)
    ap.add_argument("--skip_val_if_exists", type=int, default=1, help="1=若已有 val 則跳過切分")
    ap.add_argument("--valsync", type=int, default=1, help="1=移除 train 與 val 重疊（含 chips）")

    # chips（在 val split 之後才會進行）
    ap.add_argument("--make_chips", action="store_true")
    ap.add_argument("--chip_size", type=int, default=1024)
    ap.add_argument("--chips_per_img", type=int, default=3)
    ap.add_argument("--chip_jitter", type=float, default=0.15)
    ap.add_argument("--chip_min_area", type=int, default=4)
    ap.add_argument("--chip_format", choices=["jpg","png"], default="jpg")
    ap.add_argument("--jpg_quality", type=int, default=90)
    ap.add_argument("--png_compression", type=int, default=1)
    ap.add_argument("--clear_old_chips", type=int, default=1, help="1=重建 chips 前先清空舊檔")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    img_tr = dst/"images/train"
    lbl_tr = dst/"labels/train"
    img_te = dst/"images/test"
    for d in [img_tr, lbl_tr, img_te]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) train 影像 + 標註轉換（並行）
    train_imgs = sorted((src/"train").glob("*.png"))
    print(f"[*] converting train (images+labels) with {args.workers} workers ... total={len(train_imgs)}")
    fails=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_convert_one, src/"train", img_tr, lbl_tr, p) for p in train_imgs]
        for i,f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r: fails.append(r)
            if i % 200 == 0 or i == len(train_imgs):
                print(f"  - progress {i}/{len(train_imgs)} (fails={len(fails)})", flush=True)
    if fails:
        print(f"[warn] failed to convert {len(fails)} images (first 5): {fails[:5]}")

    # 2) test 影像複製（並行）
    test_imgs = sorted((src/"test").glob("*.png"))
    print(f"[*] copying test images with {args.workers} workers ... total={len(test_imgs)}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(shutil.copy2, p, img_te/p.name) for p in test_imgs]
        for i,_ in enumerate(as_completed(futs), 1):
            if i % 300 == 0 or i == len(test_imgs):
                print(f"  - progress {i}/{len(test_imgs)}", flush=True)

    # 3) 切 val（若需要）
    moved = _split_val_if_needed(
        dst,
        ratio=float(args.val_ratio),
        seed=int(args.val_seed),
        skip_if_exists=bool(int(args.skip_val_if_exists))
    )

    # 4) ValSync（移除 train 與 val 重疊）
    if bool(int(args.valsync)):
        _valsync_remove_overlaps(dst)

    # 5) 依「目前的 train」再生 chips（避免洩漏到 val）
    include_chips=False
    if args.make_chips:
        made = _build_chips_after_val(
            dst,
            chip_size=args.chip_size,
            chips_per_img=args.chips_per_img,
            jitter=args.chip_jitter,
            min_area=args.chip_min_area,
            fmt=args.chip_format,
            jpg_q=args.jpg_quality,
            png_c=args.png_compression,
            seed=args.seed,
            clear_old=bool(int(args.clear_old_chips)),
            workers=args.workers
        )
        include_chips = (made > 0)

    # 6) 寫 data.yaml（val=images/val；train 若 chips 存在就合併）
    names=[s.strip() for s in args.names.split(",")]
    _write_data_yaml(args.yaml, dst, names, include_chips)

if __name__ == "__main__":
    main()
