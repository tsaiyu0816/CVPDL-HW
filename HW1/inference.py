# inference.py
"""
用訓練好的權重在 data/test/img 做推論，輸出 submission.csv
格式：Image_ID,PredictionString
PredictionString = "score x y w h cls ..." 以空白分隔，多個框串起來
座標單位：像素，(x,y) 為左上角，(w,h) 為寬高；cls 固定 0（只有「豬」一類）
"""
from pathlib import Path
import argparse
import pandas as pd
from ultralytics import YOLO
from utils import read_image_hw, xyxy_to_tlwh, clamp_box, format_pred_line, natural_int
from typing import Union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/pig/yolov8s-baseline/weights/best.pt", type=str)
    ap.add_argument("--test_dir", default="data/test/img", type=str)
    ap.add_argument("--out_csv", default="submission.csv", type=str)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--conf", default=0.4, type=float)
    ap.add_argument("--iou", default=0.65, type=float)
    ap.add_argument("--device", default=0, type=str)
    ap.add_argument("--max_det", default=300, type=int)
    ap.add_argument("--tta", action="store_true", help="開啟測試時增強 (flip/multi-scale)")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    img_paths = [p for p in test_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]]
    # 依照數字 id 排序（避免評測順序問題）
    img_paths.sort(key=lambda p: (natural_int(p.stem) is None, natural_int(p.stem), p.stem))

    model = YOLO(args.weights)

    rows = []
    for img_path in img_paths:
        W, H = read_image_hw(img_path)
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            augment=args.tta,
            verbose=False
        )[0]

        pred_str_parts = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                x, y, w, h = xyxy_to_tlwh(x1, y1, x2, y2)
                x, y, w, h = clamp_box(x, y, w, h, W, H)
                pred_str_parts.append(format_pred_line(float(sc), x, y, w, h, int(c)))
        # 組 PredictionString（若無偵測，留空字串也可）
        prediction_string = " ".join(pred_str_parts)
        # Image_ID 取檔名（去副檔名），通常是數字
        rows.append({"Image_ID": img_path.stem, "PredictionString": prediction_string})

    df = pd.DataFrame(rows, columns=["Image_ID", "PredictionString"])
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    main()
