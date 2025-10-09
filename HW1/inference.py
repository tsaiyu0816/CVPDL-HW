# inference.py
"""
多權重（multi-seed / multi-run）+ 多尺度 + HFlip 的 WBF 推論器
- 支援一次丟入多個 .pt 權重（逗號/空白分隔、萬用字元、或資料夾自動搜 *.pt）
- 針對每張影像，對每個模型跑：
    * 原圖
    * （可選）HFlip
    * （可選）多尺度 imgsz 列表
  然後把所有路徑的框丟進 WBF 融合，最後再做 final_conf/Top-K 過濾。
- 輸出 submission.csv（Image_ID, PredictionString），格式同你原本規格：
  "score x y w h cls ..."，座標像素，cls 固定 0（只有「豬」一類）
"""
from pathlib import Path
import argparse
from typing import List, Tuple, Sequence
import numpy as np
import pandas as pd
import cv2
import glob
import re

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


def _expand_weights_arg(weights_arg: str) -> List[Path]:
    """
    解析 --weights：
      - 逗號或空白分隔的多個路徑/萬用字元
      - 若為資料夾，自動尋找底下的 *.pt / *.pth
      - 若為萬用字元（含 * ?），以 glob 展開
    回傳不重複、存在的檔案路徑（保持出現順序）
    """
    if not weights_arg:
        return []
    tokens = [t for t in re.split(r"[,\s]+", weights_arg) if t.strip()]
    seen = set()
    out: List[Path] = []
    for tok in tokens:
        tok = tok.strip()
        paths: List[str] = []
        p = Path(tok)
        if any(ch in tok for ch in "*?[]"):
            paths = glob.glob(tok, recursive=True)
        elif p.is_dir():
            paths = glob.glob(str(p / "**" / "*.pt"), recursive=True) + \
                    glob.glob(str(p / "**" / "*.pth"), recursive=True)
        elif p.is_file():
            paths = [str(p)]
        else:
            # 嘗試當作資料夾下通配
            paths = glob.glob(str(p / "**" / "*.pt"), recursive=True) + \
                    glob.glob(str(p / "**" / "*.pth"), recursive=True)
        for s in sorted(set(paths)):
            ps = Path(s)
            try:
                rp = ps.resolve()
            except Exception:
                rp = ps
            if rp.exists() and rp.suffix.lower() in [".pt", ".pth"]:
                key = str(rp)
                if key not in seen:
                    seen.add(key)
                    out.append(rp)
    return out


def run_single_wbf_ensemble(
    models: Sequence[YOLO],
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
    針對單張影像，對「多個模型 × (原圖 + 可選 HFlip + 可選多尺度)」做推論，最後用 WBF 融合。
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except Exception as e:
        raise RuntimeError("需要套件 ensemble-boxes：請先 `pip install ensemble-boxes`") from e

    # 讀圖尺寸；HFlip 會需要影像內容
    W, H = read_image_hw(img_path)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # 供 HFlip 用
    img_flip = cv2.flip(img_bgr, 1) if (use_hflip and img_bgr is not None) else None

    boxes_list, scores_list, labels_list = [], [], []

    # 對每個模型執行
    for model in models:
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

        # 2) HFlip
        if img_flip is not None:
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
            xyxyf = _xyxy_hflip(xyxyf, W)
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

    # 4) WBF 融合
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=wbf_iou,
        skip_box_thr=conf,        # 進入 WBF 的分數門檻
        conf_type=wbf_conf_type,  # "max" 對集成較穩
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
    # ✅ 現在 --weights 支援：逗號/空白分隔、萬用字元、或資料夾（自動找 *.pt）
    ap.add_argument(
        "--weights",
        default="runs/pig/y11x_s*/weights/best.pt",
        type=str,
        help="多個模型權重：用逗號或空白分隔，或用萬用字元，或給資料夾（自動遞迴找 *.pt）",
    )
    ap.add_argument("--test_dir", default="data/test/images", type=str)
    ap.add_argument("--out_csv", default="submission.csv", type=str)
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--conf", default=0.18, type=float, help="Ultralytics 預測與 WBF 入口分數門檻")
    ap.add_argument("--iou", default=0.65, type=float, help="Ultralytics NMS IoU")
    ap.add_argument("--device", default="0", type=str)
    ap.add_argument("--max_det", default=300, type=int)

    # 內建 TTA（非 WBF）：只在 --wbf 未啟用時才會用到
    ap.add_argument("--tta", action="store_true", help="Ultralytics 內建 TTA（非 WBF）")

    # WBF 相關
    ap.add_argument("--wbf", action="store_true", help="啟用 WBF（會改用自訂 TTA/集成流程）")
    ap.add_argument("--wbf_hflip", action="store_true", help="WBF 時加水平翻轉 TTA")
    ap.add_argument(
        "--wbf_scales",
        type=str,
        default="1.10,1.0,0.90",
        help="WBF 的多尺度倍率，逗號分隔，例如 '1.10,1.0,0.90'；會乘上 --imgsz 取整，下限 320",
    )
    ap.add_argument("--wbf_iou", type=float, default=0.60, help="WBF 內部 IoU 門檻")
    ap.add_argument("--wbf_conf_type", type=str, default="max", choices=["avg", "max"])

    # 最終輸出前的過濾
    ap.add_argument(
        "--final_conf",
        type=float,
        default=0.10,
        help="寫入 CSV 前的最終分數門檻；未指定則沿用 --conf",
    )
    ap.add_argument("--topk", type=int, default=300, help="每張圖最多保留的框數（CSV 前截斷）")
    return ap.parse_args()


def _make_models(weight_paths: List[Path]) -> List[YOLO]:
    if not weight_paths:
        raise FileNotFoundError("找不到任何可用的權重檔（請檢查 --weights 參數）")
    models: List[YOLO] = []
    print("[INFO] 將載入以下權重做集成：")
    for p in weight_paths:
        print("   -", str(p))
        models.append(YOLO(str(p)))
    return models


def main():
    args = parse_args()

    # 解析測試影像列表
    test_dir = Path(args.test_dir)
    img_paths = [p for p in test_dir.iterdir()
                 if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    img_paths.sort(key=lambda p: (natural_int(p.stem) is None, natural_int(p.stem), p.stem))

    # 展開多權重並載入模型
    weight_paths = _expand_weights_arg(args.weights)
    models = _make_models(weight_paths)

    # 解析 WBF 多尺度
    try:
        ratios = [float(s.strip()) for s in args.wbf_scales.split(",")]
    except Exception:
        ratios = [1.0]

    # 以各模型的 stride 取最大值，確保符合所有模型的 stride 對齊
    stride = 32
    try:
        strides = []
        for m in models:
            try:
                s = int(max(m.model.stride).item())
                strides.append(s)
            except Exception:
                pass
        if strides:
            stride = max(strides)
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
            xyxy, confs, clss = run_single_wbf_ensemble(
                models=models,
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
            # 非 WBF：只用第一個模型，可選 Ultralytics 內建 TTA
            res = models[0].predict(
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
