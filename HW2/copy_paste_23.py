#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, random, yaml, os
from pathlib import Path
import cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

cv2.setNumThreads(0)  # 避免和多行程互搶

# ============ utils ============
def read_labels(p: Path):
    out=[]
    if not p.exists(): return out
    for ln in p.read_text().splitlines():
        t=ln.strip().split()
        if len(t)<5: continue
        try:
            c=int(float(t[0])); cx,cy,w,h=map(float,t[1:5])
            out.append([c,cx,cy,w,h])
        except: pass
    return out

def write_labels(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for c,cx,cy,w,h in rows:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def find_image(img_dir: Path, stem: str):
    for ext in (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"):
        q=img_dir/f"{stem}{ext}"
        if q.exists(): return q
    return None

def xywhn_to_xyxy_abs(cx,cy,w,h,W,H):
    x1=(cx-w/2)*W; y1=(cy-h/2)*H; x2=(cx+w/2)*W; y2=(cy+h/2)*H
    return [x1,y1,x2,y2]

def xyxy_abs_to_xywhn(x1,y1,x2,y2,W,H):
    cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
    w=(x2-x1)/W; h=(y2-y1)/H
    return [cx,cy,w,h]

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def rand_scale(minr,maxr): return minr + (maxr-minr)*random.random()

def alpha_blend(dst, patch, x1, y1, blur_frac=0.12):
    h,w=patch.shape[:2]
    roi=dst[y1:y1+h, x1:x1+w]
    if roi.shape[0] != h or roi.shape[1] != w:  # 越界保護
        return False
    mask=np.ones((h,w),np.uint8)*255
    k=int(max(3, round(min(h,w)*blur_frac)))
    if k%2==0: k+=1
    mask=cv2.GaussianBlur(mask,(k,k),0)
    a=(mask/255.0)[:,:,None]
    roi[:]= (roi*(1-a)+patch*a).astype(np.uint8)
    return True
def road_mask_hsv(img, s_th=60, v_th=60, green_gap=15):
    """簡易『像馬路』遮罩：低飽和(灰)、夠亮、且不是偏綠(草地)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    grayish = (S < s_th) & (V > v_th)
    greenish = (img[:,:,1].astype(int) - img[:,:,2].astype(int) > green_gap)  # G-R
    m = (grayish & (~greenish)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m  # 0/255

def color_match_to_bg(patch, bg_roi):
    """把貼片顏色對齊到當地背景的均值/方差，降低『貼上去很突兀』"""
    p = patch.astype(np.float32); b = bg_roi.astype(np.float32)
    for c in range(3):
        mp, sp = p[...,c].mean(), p[...,c].std()+1e-6
        mb, sb = b[...,c].mean(), b[...,c].std()+1e-6
        p[...,c] = (p[...,c]-mp)*(sb/sp)+mb
    return np.clip(p,0,255).astype(np.uint8)


# ============ globals for workers ============
G = {
    "donors": None,   # list of (img_path_str, x1,y1,x2,y2, class_id)
    "out_img": None,  # str path
    "out_lbl": None,  # str path
}

def _init_worker(donors, out_img, out_lbl):
    G["donors"] = donors
    G["out_img"] = out_img
    G["out_lbl"] = out_lbl


def _work_one(task):
    rng = random.Random(task["seed"])
    imgp = Path(task["imgp"])
    im = cv2.imread(str(imgp), cv2.IMREAD_COLOR)
    if im is None:
        return (0, 0)
    H, W = im.shape[:2]

    # 建立一次『像馬路』遮罩（可選）
    road_m = None
    if task.get("road_only", False):
        road_m = road_mask_hsv(
            im,
            s_th=task.get("road_s", 60),
            v_th=task.get("road_v", 60),
            green_gap=task.get("green_gap", 15),
        )

    # 既有標註
    existing = []
    for c, cx, cy, w, h in task["rows"]:
        x1, y1, x2, y2 = xywhn_to_xyxy_abs(cx, cy, w, h, W, H)
        existing.append([x1, y1, x2, y2])
    new_rows = list(task["rows"])

    adds = 0; tries = 0
    per_img = task["per_img"]
    donors = G["donors"]

    while adds < per_img and tries < per_img * 30:
        tries += 1
        dimg, dx1, dy1, dx2, dy2, dc = donors[rng.randrange(len(donors))]
        dim = cv2.imread(dimg, cv2.IMREAD_COLOR)
        if dim is None:
            continue
        patch = dim[int(dy1):int(dy2), int(dx1):int(dx2)]
        if patch.size == 0:
            continue

        # flip + scale
        if rng.random() < 0.5:
            patch = cv2.flip(patch, 1)
        ph, pw = patch.shape[:2]
        scale = rand_scale(task["scale_min"], task["scale_max"])
        nh, nw = max(2, int(ph * scale)), max(2, int(pw * scale))
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 占比限制
        if (nw / W) * (nh / H) > task["place_area_max"]:
            continue

        # 找位置：避免與既有框重疊過大 + 路面檢查
        ok = False
        for _ in range(40):
            x1 = rng.randint(0, max(0, W - nw))
            y1 = rng.randint(0, max(0, H - nh))
            x2, y2 = x1 + nw, y1 + nh
            cand = [x1, y1, x2, y2]

            # IoU 檢查
            if not all(iou(cand, e) < 0.20 for e in existing):
                continue

            # 路面比例檢查（可選）
            if road_m is not None:
                roi_m = road_m[y1:y2, x1:x2]
                if roi_m.size == 0 or roi_m.mean() < task["road_ratio"] * 255.0:
                    continue

            ok = True
            break
        if not ok:
            continue

        # 局部顏色匹配（可選）
        if task.get("color_match", False):
            patch = color_match_to_bg(patch, im[y1:y2, x1:x2])

        # 貼上
        if not alpha_blend(im, patch, x1, y1, blur_frac=0.10):
            continue

        # 新標註
        cx, cy, w, h = xyxy_abs_to_xywhn(x1, y1, x2, y2, W, H)
        new_rows.append([dc, cx, cy, w, h])
        existing.append([x1, y1, x2, y2])
        adds += 1

    if adds == 0:
        return (0, 0)

    stem = f"{imgp.stem}__cp"
    out_img = Path(G["out_img"]) / f"{stem}.jpg"
    out_lbl = Path(G["out_lbl"]) / f"{stem}.txt"
    cv2.imwrite(str(out_img), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    write_labels(out_lbl, new_rows)
    return (1, adds)


# ============ main ============
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="YOLO root, e.g. data/hw2_yolo")
    ap.add_argument("--yaml", required=True, help="data yaml, e.g. data/hw2_fold0.yaml")
    ap.add_argument("--classes", default="2,3", help="target class ids, e.g. '2,3'")
    ap.add_argument("--per_img", type=int, default=3, help="max pastes per target image")
    ap.add_argument("--donor_area_min", type=float, default=0.003, help="min normalized area of donor box")
    ap.add_argument("--donor_area_max", type=float, default=0.15, help="max normalized area of donor box")
    ap.add_argument("--place_area_max", type=float, default=0.20, help="limit pasted box area on target (normalized)")
    ap.add_argument("--scale_min", type=float, default=0.8)
    ap.add_argument("--scale_max", type=float, default=1.3)
    ap.add_argument("--prefer_bg", action="store_true", help="prefer target images that lack target classes")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--cp_only", action="store_true", help="use only images/train_cp for training")
    ap.add_argument("--road_only", action="store_true", help="只貼在像馬路的區域")
    ap.add_argument("--road_ratio", type=float, default=0.6, help="候選區域需有多少比例屬於『路面』")
    ap.add_argument("--road_s", type=int, default=60)
    ap.add_argument("--road_v", type=int, default=60)
    ap.add_argument("--green_gap", type=int, default=15)
    ap.add_argument("--tighten", type=float, default=0.08, help="收縮donor bbox的比例，去掉邊緣背景")
    ap.add_argument("--color_match", action="store_true", help="貼上前做局部顏色匹配")
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    root=Path(args.root)
    img_tr=root/"images/train"; lbl_tr=root/"labels/train"
    out_img=root/"images/train_cp"; out_lbl=root/"labels/train_cp"
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

    targets={int(x) for x in args.classes.replace(","," ").split() if x.strip()}

    # 1) collect donors (只記路徑 + bbox + 類別，避免巨量影像進行程間傳遞)
    donors=[]
    for lbp in sorted(lbl_tr.glob("*.txt")):
        rows=read_labels(lbp)
        if not rows: continue
        imgp=find_image(img_tr, lbp.stem)
        if imgp is None: continue
        im=cv2.imread(str(imgp), cv2.IMREAD_COLOR)
        if im is None: continue
        H,W=im.shape[:2]
        for c,cx,cy,w,h in rows:
            if c not in targets: continue
            area=w*h
            if area<args.donor_area_min or area>args.donor_area_max: 
                continue
            x1,y1,x2,y2=xywhn_to_xyxy_abs(cx,cy,w,h,W,H)
            # tighten bbox 8%（可調），降低把原地面一起帶進來
            t = float(args.tighten)
            if t > 1e-6:
                dx = (x2 - x1) * t * 0.5
                dy = (y2 - y1) * t * 0.5
                x1 += dx; x2 -= dx; y1 += dy; y2 -= dy

            donors.append((str(imgp), x1,y1,x2,y2, c))
    print(f"[cp] donors collected: {len(donors)}")
    if not donors:
        print("[cp] no donors found; relax donor_area_min/max or check class ids.")
        return

    # 2) target list
    label_files=sorted(lbl_tr.glob("*.txt"))
    targets_abs=[]
    for lbp in label_files:
        rows=read_labels(lbp)
        imgp=find_image(img_tr, lbp.stem)
        if imgp is None: continue
        has_t = any(int(r[0]) in targets for r in rows)
        targets_abs.append((str(imgp), rows, has_t))
    if args.prefer_bg:
        targets_abs.sort(key=lambda x: (x[2],))  # 先處理沒有 2/3 的

    # 3) 並行處理
    
    made_total=0; pasted_total=0
    with ProcessPoolExecutor(max_workers=int(args.workers),
                            initializer=_init_worker,
                            initargs=(donors, str(out_img), str(out_lbl))) as ex:
        futures=[]
        for i, (imgp_str, rows, _) in enumerate(targets_abs):
            task = dict(
                imgp=imgp_str,
                rows=rows,
                per_img=int(args.per_img),
                place_area_max=float(args.place_area_max),
                scale_min=float(args.scale_min),
                scale_max=float(args.scale_max),
                seed=int(args.seed) + i,
                # ↓↓↓ 新增這些 ↓↓↓
                road_only=bool(args.road_only),
                road_ratio=float(args.road_ratio),
                road_s=int(args.road_s),
                road_v=int(args.road_v),
                green_gap=int(args.green_gap),
                color_match=bool(args.color_match),
            )
            futures.append(ex.submit(_work_one, task))


        for k, fu in enumerate(as_completed(futures), 1):
            made, adds = fu.result()
            made_total += made; pasted_total += adds
            if k % 100 == 0 or k == len(futures):
                print(f"[cp][{k}/{len(futures)}] made={made_total}, pasted={pasted_total}")

    print(f"[cp] augmented images written: {made_total}, pasted instances: {pasted_total}")

    # 4) 更新 YAML
    ypath = Path(args.yaml)
    y = yaml.safe_load(ypath.read_text())

    cp_img_dir = root / "images/train_cp"
    cp_lbl_dir = root / "labels/train_cp"

    if args.cp_only:
        # 安全檢查：確保 CP 目錄不是空的（同時檢查影像與標註）
        if not (cp_img_dir.exists() and any(cp_img_dir.iterdir())
                and cp_lbl_dir.exists() and any(cp_lbl_dir.glob("*.txt"))):
            raise SystemExit("[cp] images/labels/train_cp 為空，請先生成 CP 再切換成 --cp_only。")
        y["path"] = str(root)
        y["train"] = "images/train_cp"                     # 只用 CP 圖
    else:
        train_field = y.get("train", "images/train")
        train_list = [train_field] if isinstance(train_field, str) else list(train_field)
        if "images/train_cp" not in train_list:
            train_list.append("images/train_cp")           # 原圖 + CP 圖
        y["path"] = str(root)
        y["train"] = train_list

    ypath.write_text(yaml.safe_dump(y, sort_keys=False))
    print(f"[cp] YAML updated: train={y['train']}")
    print(f"[hint] 清掉快取：rm -f {root}/labels/train.cache")


if __name__=="__main__":
    main()
