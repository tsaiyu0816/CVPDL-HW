#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, time
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2
import torch

# 可選：SAHI（僅在 --mode sahi 時才用到）
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# Ultralytics
from ultralytics import YOLO


# ---------------- Basics ----------------
def parse_pc_conf(s: str):
    """'0:0.05,1:0.03' -> {0:0.05, 1:0.03}"""
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        out[int(k)] = float(v)
    return out

def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    union = aw * ah + bw * bh - inter + 1e-9
    return inter / union

def wbf_merge(boxes: List[Tuple[float, float, float, float]],
              scores: List[float], iou_thr: float):
    """簡化版 WBF，輸入/輸出皆為 xywh（浮點）。"""
    if not boxes:
        return [], []
    order = np.argsort(-np.asarray(scores))
    boxes = [boxes[i] for i in order]
    scores = [scores[i] for i in order]
    used = [False] * len(boxes)
    out_boxes, out_scores = [], []
    for i, (b, s) in enumerate(zip(boxes, scores)):
        if used[i]:
            continue
        grp_idx = [i]; used[i] = True
        wsum = s
        cx, cy, cw, ch = b
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou_xywh((cx, cy, cw, ch), boxes[j]) >= iou_thr:
                used[j] = True; grp_idx.append(j)
                wsum_new = wsum + scores[j]
                cx = (cx * wsum + boxes[j][0] * scores[j]) / wsum_new
                cy = (cy * wsum + boxes[j][1] * scores[j]) / wsum_new
                cw = (cw * wsum + boxes[j][2] * scores[j]) / wsum_new
                ch = (ch * wsum + boxes[j][3] * scores[j]) / wsum_new
                wsum = wsum_new
        out_boxes.append((cx, cy, cw, ch))
        out_scores.append(float(np.mean([scores[k] for k in grp_idx])))
    return out_boxes, out_scores


# ----------- Visualization -----------
def _color_for(c:int):
    rng = np.random.default_rng(seed=12345 + int(c))
    return tuple(int(x) for x in rng.integers(80, 255, size=3))

def draw_boxes(img, items, class_names=None, thickness=2):
    H, W = img.shape[:2]
    out = img.copy()
    for (x, y, w, h), s, c in items:
        x1 = max(0, min(int(round(x)), W - 1))
        y1 = max(0, min(int(round(y)), H - 1))
        x2 = max(0, min(int(round(x + w)), W - 1))
        y2 = max(0, min(int(round(y + h)), H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color_for(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(thickness))
        name = (class_names[int(c)] if (class_names and int(c) < len(class_names)) else f"cls{int(c)}")
        label = f"{name} {float(s):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(0, min(x1, W - tw - 6))
        ty = max(th + 6, min(y1, H - 1))
        cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 6, ty), color, -1)
        cv2.putText(out, label, (tx + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ----------- Helpers for TTA -----------
def flip_boxes_h(items, W):
    out = []
    for (x, y, w, h), s, c in items:
        out.append(((W - (x + w), y, w, h), s, c))
    return out

def scale_boxes(items, scale):
    if abs(scale - 1.0) < 1e-6:
        return items
    inv = 1.0 / scale
    out = []
    for (x, y, w, h), s, c in items:
        out.append(((x * inv, y * inv, w * inv, h * inv), s, c))
    return out


# ----------- SAHI -----------
def run_sahi_on_image(image, det_model, slice_size, overlap,
                      post_type="NMS", match_thr=0.6):
    """回傳 [(xywh_float), score, cls]"""
    def _call():
        return get_sliced_prediction(
            image=image,
            detection_model=det_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type=post_type,
            postprocess_match_threshold=match_thr,
            verbose=0
        )
    try:
        pred = _call()
    except RuntimeError as e:
        msg = str(e)
        if ("Half" in msg and "float" in msg) or ("c10::Half" in msg):
            try:
                det_model.model.overrides["half"] = False
                det_model.model.overrides["fuse"] = False
            except Exception:
                pass
            pred = _call()
        else:
            raise

    out = []
    for o in pred.object_prediction_list:
        s = float(o.score.value)
        c = int(o.category.id)
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        w, h = (x2 - x1), (y2 - y1)
        if w > 0 and h > 0:
            out.append(((float(x1), float(y1), float(w), float(h)), float(s), int(c)))
    return out


# ----------- FULL-image predictor -----------
def yolov8_predict_full(model: YOLO, img: np.ndarray, imgsz: int, conf: float, iou: float, device, max_det: int):
    """
    用 Ultralytics 直接整圖推論（單次）。回傳 [(xywh_float), score, cls]
    注意：這裡的結果已過一次 NMS（Ultralytics 內建）。
    """
    rs = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou,
                       device=device, max_det=max_det, verbose=False)
    out = []
    if not rs:
        return out
    r = rs[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), s, c in zip(xyxy, scores, clss):
        w, h = (x2 - x1), (y2 - y1)
        if w > 0 and h > 0:
            out.append(((float(x1), float(y1), float(w), float(h)), float(s), int(c)))
    return out


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser("Unified inference (full-image & SAHI)")
    ap.add_argument("--mode", choices=["full","sahi"], default="full",
                    help="full=整圖（和 val 一致的 baseline）；sahi=切片融合")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test_dir", required=True)

    # 通用/寫檔
    ap.add_argument("--out_csv", default="submission.csv")
    ap.add_argument("--save_details", action="store_true")
    ap.add_argument("--viz_dir", default="")
    ap.add_argument("--viz_limit", type=int, default=0)
    ap.add_argument("--viz_every", type=int, default=0)

    # 推論門檻
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--per_class_conf", default="")
    ap.add_argument("--max_det_per_class", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--max_det", type=int, default=800)

    # TTA（兩種模式都可用）
    ap.add_argument("--tta_hflip", action="store_true")
    ap.add_argument("--tta_scales", default="1.0")  # 逗點分隔，例如 "0.8,1.0,1.2"

    # full-image 參數
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--device", default="0")  # "0" or "cpu"

    # SAHI 參數
    ap.add_argument("--slice", type=int, default=1792)
    ap.add_argument("--overlap", type=float, default=0.40)
    ap.add_argument("--post_type", default="GREEDYNMM", choices=["NMS", "GREEDYNMM", "LSNMS"])
    ap.add_argument("--post_match_thr", type=float, default=0.50)
    ap.add_argument("--wbf", action="store_true")
    ap.add_argument("--wbf_iou", type=float, default=0.50)

    args = ap.parse_args()

    # 準備資料集
    test_dir = Path(args.test_dir)
    img_paths = sorted([
        *test_dir.glob("*.png"), *test_dir.glob("*.jpg"), *test_dir.glob("*.jpeg"),
        *test_dir.glob("*.PNG"), *test_dir.glob("*.JPG"), *test_dir.glob("*.JPEG"),
    ])
    if not img_paths:
        print(f"[!] No images found under: {test_dir}")
        return

    pc_thr = parse_pc_conf(args.per_class_conf)
    scales = [float(s) for s in args.tta_scales.split(",") if s.strip()]

    # 視覺化資料夾
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # 準備模型
    if args.mode == "full":
        model = YOLO(args.weights)
        names = model.model.names if hasattr(model, "model") else (model.names if hasattr(model, "names") else None)
        device = args.device
    else:
        # SAHI
        det = UltralyticsDetectionModel(
            model_path=args.weights,
            confidence_threshold=min(args.conf, 0.07),
            device=("cuda:0" if torch.cuda.is_available() and "cuda" in str(args.device).lower() else (args.device if "cpu" in str(args.device).lower() else "cpu"))
        )
        # 建立一次 predictor
        try:
            dummy = np.zeros((max(32, args.slice), max(32, args.slice), 3), dtype=np.uint8)
            _ = det.model.predict(source=dummy, imgsz=args.slice, conf=0.01, verbose=False,
                                  device=(0 if "cuda" in str(args.device).lower() and torch.cuda.is_available() else "cpu"))
            o = det.model.overrides
            o["device"] = (0 if "cuda" in str(args.device).lower() and torch.cuda.is_available() else "cpu")
            o["half"] = True if torch.cuda.is_available() and "cuda" in str(args.device).lower() else False
            o["max_det"] = max(1000, args.max_det)
            o["iou"] = min(max(args.iou, 0.05), 0.95)  # 用於 predictor 的內部 NMS（仍建議讓 SAHI 主導融合）
            o["agnostic_nms"] = False
            o["imgsz"] = int(args.slice)
        except Exception as e:
            print(f"[Warn] override error: {e}")
        # 類別名
        names = None
        for chain in (["names"], ["model","names"], ["model","model","names"]):
            obj = det
            try:
                for a in chain:
                    obj = getattr(obj, a)
                names = obj
                break
            except Exception:
                pass
        if isinstance(names, dict):
            maxk = max(int(k) for k in names.keys())
            arr = [f"cls{i}" for i in range(maxk+1)]
            for k,v in names.items():
                arr[int(k)] = str(v)
            names = arr
        elif isinstance(names, (list,tuple)):
            names = [str(x) for x in names]
        else:
            names = None

    # 蒐集輸出
    rows_submission, rows_details = [], []
    per_image_counts = []
    per_class_counter = Counter()
    t0 = time.time(); t_im = []

    def keep_after_thr(per_cls_dict, c, s, gthr):
        """per-class 門檻（優先）+ global conf"""
        thr = per_cls_dict.get(c, gthr)
        return s >= thr

    # 逐張影像
    for idx, p in enumerate(img_paths, 1):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            rows_submission.append({"Image_ID": p.name, "PredictionString": ""})
            continue
        H, W = im.shape[:2]
        tic = time.time()
        preds = []  # [(xywh_f), score, cls]

        if args.mode == "full":
            for s in scales:
                if abs(s - 1.0) < 1e-6:
                    im_s = im
                else:
                    im_s = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)
                r1 = yolov8_predict_full(
                    model=model, img=im_s, imgsz=args.imgsz,
                    conf=min(args.conf, 0.99), iou=args.iou, device=args.device, max_det=args.max_det
                )
                r1 = scale_boxes(r1, s)  # map 回原圖比例
                preds.extend(r1)

                if args.tta_hflip:
                    im_f = cv2.flip(im_s, 1)
                    r2 = yolov8_predict_full(
                        model=model, img=im_f, imgsz=args.imgsz,
                        conf=min(args.conf, 0.99), iou=args.iou, device=args.device, max_det=args.max_det
                    )
                    r2 = flip_boxes_h(r2, im_s.shape[1])
                    r2 = scale_boxes(r2, s)
                    preds.extend(r2)

        else:  # SAHI
            for s in scales:
                if abs(s - 1.0) < 1e-6:
                    im_s = im
                else:
                    im_s = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)
                r1 = run_sahi_on_image(
                    im_s, det, args.slice, args.overlap,
                    post_type=args.post_type, match_thr=args.post_match_thr
                )
                r1 = scale_boxes(r1, s)
                preds.extend(r1)

                if args.tta_hflip:
                    im_f = cv2.flip(im_s, 1)
                    rf = run_sahi_on_image(
                        im_f, det, args.slice, args.overlap,
                        post_type=args.post_type, match_thr=args.post_match_thr
                    )
                    rf = flip_boxes_h(rf, im_s.shape[1])
                    rf = scale_boxes(rf, s)
                    preds.extend(rf)

        # 輕量預過濾
        pre_min_conf = max(0.00, min(args.conf, 0.15))
        preds = [pp for pp in preds if pp[1] >= pre_min_conf]

        # 依類別分組 →（可選）WBF → per-class 門檻 → per-class 限制
        by_cls = defaultdict(lambda: ([], []))  # {c: (boxes, scores)}
        for (xywh), s, c in preds:
            bb, ss = by_cls[c]
            bb.append(xywh); ss.append(s)

        merged_items = []  # for viz & export
        for c, (bxs, scores) in by_cls.items():
            if not bxs:
                continue
            if args.wbf and len(bxs) > 1:
                bxs, scores = wbf_merge(bxs, scores, iou_thr=args.wbf_iou)

            keep = [(b, s) for b, s in zip(bxs, scores) if keep_after_thr(pc_thr, c, s, args.conf)]
            if args.max_det_per_class and len(keep) > args.max_det_per_class:
                keep = sorted(keep, key=lambda t: -t[1])[:args.max_det_per_class]

            for (x, y, w, h), s in keep:
                merged_items.append(((x, y, w, h), s, c))

        # global top-k（依分數）
        if args.topk and merged_items:
            merged_items.sort(key=lambda t: -t[1])
            merged_items = merged_items[:args.topk]

        # 組 submission 欄位（這裡才轉成 int）
        parts = []
        for (x, y, w, h), s, c in merged_items:
            parts += [f"{float(s):.6f}", str(int(round(x))), str(int(round(y))),
                      str(int(round(w))), str(int(round(h))), str(int(c))]

            rows_details.append({
                "Image_ID": p.name,
                "class_id": int(c),
                "class_name": (names[int(c)] if names and int(c) < len(names) else f"cls{int(c)}"),
                "score": float(s),
                "x": int(round(x)), "y": int(round(y)), "w": int(round(w)), "h": int(round(h))
            })
            per_class_counter[int(c)] += 1

        rows_submission.append({"Image_ID": p.name, "PredictionString": " ".join(parts)})
        per_image_counts.append(len(merged_items))

        # 視覺化
        if viz_dir:
            save_ok = True
            if args.viz_every > 0:
                save_ok = (idx % args.viz_every == 0)
            if args.viz_limit > 0 and sum(1 for _ in viz_dir.glob("*.jpg")) >= args.viz_limit:
                save_ok = False
            if save_ok:
                viz = draw_boxes(im, merged_items, class_names=names)
                cv2.imwrite(str(viz_dir / p.with_suffix(".jpg").name), viz)

        toc = time.time()
        t_im.append(toc - tic)
        if idx % 50 == 0 or idx == len(img_paths):
            print(f"[{idx}/{len(img_paths)}] done...")

    # ------ 寫 submission.csv ------
    df = pd.DataFrame(rows_submission)
    def to_id(s: str):
        s = re.sub(r'\.(png|jpg|jpeg)$', '', s, flags=re.I)
        s = re.sub(r'^img0*', '', s, flags=re.I)
        return int(s)
    df["Image_ID"] = df["Image_ID"].map(to_id)
    df.to_csv(args.out_csv, index=False)
    print(f"[✓] wrote {args.out_csv} | images={len(df)}")

    # ------ 詳細偵測 ------
    if args.save_details and rows_details:
        dname = Path(args.out_csv).with_suffix("").as_posix() + "_dets.csv"
        df_det = pd.DataFrame(rows_details)
        df_det["ImageNum"] = df_det["Image_ID"].map(lambda s: to_id(s))
        df_det.to_csv(dname, index=False)
        print(f"[✓] wrote {dname} | dets={len(df_det)}")

    # ------ 簡要統計 ------
    total_time = time.time() - t0
    avg_im = float(np.mean(t_im)) if t_im else 0.0
    fps = (1.0 / avg_im) if avg_im > 0.0 else 0.0
    n_img = int(len(img_paths))
    n_det = int(sum(int(x) for x in per_image_counts))
    avg_det = float(n_det) / max(1, n_img)

    stat_lines = []
    stat_lines.append(f"Images: {n_img}")
    stat_lines.append(f"Total dets: {n_det} | avg dets/img: {avg_det:.2f}")
    stat_lines.append(f"Latency: {avg_im*1000.0:.1f} ms/img | FPS: {fps:.2f}")
    stat_lines.append(f"Total wall time: {float(total_time):.2f} s")
    stat_lines.append("Per-class counts:")
    classes_sorted = sorted(per_class_counter.keys())
    for c in classes_sorted:
        name = (names[int(c)] if (names and int(c) < len(names)) else f"cls{int(c)}")
        stat_lines.append(f"  - {int(c):>2} ({name}): {int(per_class_counter[c])}")

    stat_path = Path(args.out_csv).with_suffix("").as_posix() + "_stats.txt"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines) + "\n")
    print("[✓] wrote", stat_path)
    print("---- Summary ----")
    print("\n".join(stat_lines))


if __name__ == "__main__":
    main()
