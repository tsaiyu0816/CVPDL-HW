# inference.py
"""
用訓練好的權重在 data/test/images 做推論，輸出 submission.csv
格式：Image_ID,PredictionString
PredictionString = "score x y w h cls ..." 以空白分隔，多個框串起來
座標單位：像素，(x,y) 為左上角，(w,h) 為寬高；cls 固定 0（只有「豬」一類）
"""
from pathlib import Path
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2

from ultralytics import YOLO
from utils import read_image_hw, xyxy_to_tlwh, clamp_box, format_pred_line, natural_int

EPS = 1e-6


def _collect_xyxy_scores_cls(res):
    """從 Ultralytics 預測結果取出 xyxy/score/cls（numpy）"""
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)
    xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(float)
    confs = res.boxes.conf.detach().cpu().numpy().astype(float)
    clss = res.boxes.cls.detach().cpu().numpy().astype(int)
    return xyxy, confs, clss


def _xyxy_hflip(xyxy: np.ndarray, W: int) -> np.ndarray:
    """把水平翻轉影像的 xyxy 轉回原圖座標"""
    if xyxy.size == 0:
        return xyxy
    out = xyxy.copy()
    x1 = xyxy[:, 0].copy()
    x2 = xyxy[:, 2].copy()
    out[:, 0] = W - x2
    out[:, 2] = W - x1
    return out


def sanitize_and_norm_xyxy(xyxy_px: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    像素 xyxy -> 乾淨的 [0,1] xyxy；保證 x2>x1, y2>y1，避免 WBF 零面積警告。
    """
    if xyxy_px.size == 0:
        return xyxy_px
    b = xyxy_px.astype(np.float32).copy()
    # 夾進圖內，預留 EPS，並保證正面積
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, W - EPS)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, H - EPS)
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + EPS)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + EPS)
    # 轉成 normalized 並再夾一次
    b[:, [0, 2]] /= float(W)
    b[:, [1, 3]] /= float(H)
    return np.clip(b, EPS, 1.0 - EPS)


def _from_norm_xyxy(xyxy: np.ndarray, W: int, H: int) -> np.ndarray:
    if xyxy.size == 0:
        return xyxy
    out = xyxy.astype(np.float32).copy()
    out[:, [0, 2]] *= float(W)
    out[:, [1, 3]] *= float(H)
    return out


def run_single_wbf(
    model: YOLO,
    img_path: Path,
    imgsz: int,
    conf: float,
    iou_pred: float,
    device: str,
    max_det: int,
    use_hflip: bool,
    scale_list: List[int],
    wbf_iou: float,
    wbf_conf_type: str,
):
    """
    針對單張影像：
      - 原圖推論
      - 可選：水平翻轉推論（再映回）
      - 可選：多尺度推論（改 imgsz）
      - 用 WBF 融合上述多路輸出
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except Exception as e:
        raise RuntimeError("需要套件 ensemble-boxes：請先 `pip install ensemble-boxes`") from e

    W, H = read_image_hw(img_path)

    boxes_list, scores_list, labels_list = [], [], []

    # 1) 原圖
    res0 = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou_pred,
        device=device,
        max_det=max_det,
        augment=False,
        verbose=False,
    )[0]
    xyxy0, sc0, cl0 = _collect_xyxy_scores_cls(res0)
    boxes_list.append(sanitize_and_norm_xyxy(xyxy0, W, H).tolist())
    scores_list.append(sc0.tolist())
    labels_list.append(cl0.tolist())

    # 2) 水平翻轉
    if use_hflip:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        if img is not None:
            img_flip = cv2.flip(img, 1)
            resf = model.predict(
                source=img_flip,
                imgsz=imgsz,
                conf=conf,
                iou=iou_pred,
                device=device,
                max_det=max_det,
                augment=False,
                verbose=False,
            )[0]
            xyxyf, scf, clf = _collect_xyxy_scores_cls(resf)
            xyxyf = _xyxy_hflip(xyxyf, W)  # 映回原圖座標
            boxes_list.append(sanitize_and_norm_xyxy(xyxyf, W, H).tolist())
            scores_list.append(scf.tolist())
            labels_list.append(clf.tolist())

    # 3) 多尺度
    for s in scale_list:
        if s == imgsz:
            continue
        res_s = model.predict(
            source=str(img_path),
            imgsz=s,
            conf=conf,
            iou=iou_pred,
            device=device,
            max_det=max_det,
            augment=False,
            verbose=False,
        )[0]
        xyxys, scs, cls_ = _collect_xyxy_scores_cls(res_s)
        boxes_list.append(sanitize_and_norm_xyxy(xyxys, W, H).tolist())
        scores_list.append(scs.tolist())
        labels_list.append(cls_.tolist())

    # 若完全沒框
    if sum(len(b) for b in boxes_list) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)

    # 4) WBF
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=wbf_iou,          # WBF 用的 IoU
        skip_box_thr=conf,        # 進入 WBF 的分數門檻
        conf_type=wbf_conf_type,  # "max" 比較穩
        allows_overflow=False,
    )

    # 還原像素座標並夾到影像範圍
    xyxy_pix = _from_norm_xyxy(np.asarray(wbf_boxes, dtype=float), W, H)
    xyxy_pix[:, 0] = np.clip(xyxy_pix[:, 0], 0, W - EPS)
    xyxy_pix[:, 2] = np.clip(xyxy_pix[:, 2], 0, W - EPS)
    xyxy_pix[:, 1] = np.clip(xyxy_pix[:, 1], 0, H - EPS)
    xyxy_pix[:, 3] = np.clip(xyxy_pix[:, 3], 0, H - EPS)

    scores = np.asarray(wbf_scores, dtype=float)
    labels = np.asarray(wbf_labels, dtype=int)

    # 排序/截斷
    if len(scores) > 0:
        order = np.argsort(-scores)[:max_det]
        xyxy_pix, scores, labels = xyxy_pix[order], scores[order], labels[order]

    return xyxy_pix, scores, labels


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/pig/yolov8s-baseline/weights/best.pt", type=str)
    ap.add_argument("--test_dir", default="data/test/images", type=str)
    ap.add_argument("--out_csv", default="submission.csv", type=str)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--conf", default=0.4, type=float)  # 預測與 WBF 入口閥值
    ap.add_argument("--iou", default=0.65, type=float)  # Ultralytics NMS IoU
    ap.add_argument("--device", default=0, type=str)
    ap.add_argument("--max_det", default=300, type=int)

    # 內建 TTA（非 WBF）
    ap.add_argument("--tta", action="store_true", help="Ultralytics 內建 TTA（非 WBF）")

    # WBF 相關
    ap.add_argument("--wbf", action="store_true", help="啟用 WBF（會改用自訂 TTA 流程）")
    ap.add_argument("--wbf_hflip", action="store_true", help="WBF 時加水平翻轉 TTA")
    ap.add_argument("--wbf_scales", type=str, default="1.0",
                    help="WBF 的多尺度倍率，逗號分隔，例如 '0.95,1.0'；會乘上 --imgsz 取整，下限 320")
    ap.add_argument("--wbf_iou", type=float, default=0.60, help="WBF 內部 IoU 門檻")
    ap.add_argument("--wbf_conf_type", type=str, default="max", choices=["avg", "max"])

    # 最終輸出前的過濾
    ap.add_argument("--final_conf", type=float, default=None,
                    help="寫入 CSV 前的最終分數門檻；未指定則沿用 --conf")
    ap.add_argument("--topk", type=int, default=300, help="每張圖最多保留的框數（CSV 前截斷）")
    return ap.parse_args()


def main():
    args = parse_args()

    test_dir = Path(args.test_dir)
    img_paths = [p for p in test_dir.iterdir()
                 if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    # 依數字 id 排序（避免評測順序問題）
    img_paths.sort(key=lambda p: (natural_int(p.stem) is None, natural_int(p.stem), p.stem))

    model = YOLO(args.weights)

    # 解析 WBF 多尺度
    try:
        ratios = [float(s.strip()) for s in args.wbf_scales.split(",")]
    except Exception:
        ratios = [1.0]
    stride = 32
    try:
        stride = int(max(model.model.stride).item())
    except Exception:
        pass


    scale_list = []
    for r in ratios:
        s = int(round(args.imgsz * r))
        s = max(320, int(round(s / stride) * stride))  # 量化成 stride 的倍數（四捨五入）
        scale_list.append(s)
    scale_list = sorted(set(scale_list))

    rows = []
    final_thr = args.final_conf if args.final_conf is not None else args.conf

    for img_path in img_paths:
        W, H = read_image_hw(img_path)

        if args.wbf:
            xyxy, confs, clss = run_single_wbf(
                model=model,
                img_path=img_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou_pred=args.iou,
                device=args.device,
                max_det=args.max_det,
                use_hflip=args.wbf_hflip,
                scale_list=scale_list,
                wbf_iou=args.wbf_iou,
                wbf_conf_type=args.wbf_conf_type,
            )
        else:
            # 原本的單路推論（可選內建 TTA，但非 WBF）
            res = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                augment=args.tta,
                verbose=False,
            )[0]
            xyxy, confs, clss = _collect_xyxy_scores_cls(res)

        # 最終過濾（WBF / 非 WBF 一致）
        if confs is not None and len(confs) > 0:
            keep = confs >= final_thr
            xyxy, confs, clss = xyxy[keep], confs[keep], clss[keep]
            if len(confs) > 0:
                order = np.argsort(-confs)
                if args.topk is not None:
                    order = order[:args.topk]
                xyxy, confs, clss = xyxy[order], confs[order], clss[order]

        # 組 PredictionString（若無偵測，留空字串）
        pred_str_parts = []
        if xyxy is not None and len(xyxy) > 0:
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                x, y, w, h = xyxy_to_tlwh(x1, y1, x2, y2)
                x, y, w, h = clamp_box(x, y, w, h, W, H)
                pred_str_parts.append(format_pred_line(float(sc), x, y, w, h, int(c)))
        prediction_string = " ".join(pred_str_parts)
        rows.append({"Image_ID": img_path.stem, "PredictionString": prediction_string})

    df = pd.DataFrame(rows, columns=["Image_ID", "PredictionString"])
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(df)} rows.")


if __name__ == "__main__":
    main()
