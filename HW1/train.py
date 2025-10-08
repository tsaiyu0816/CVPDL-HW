#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import re
from pathlib import Path
import urllib.request, os
from ultralytics.utils.checks import check_file
from ultralytics import YOLO

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")

def parse_args():
    p = argparse.ArgumentParser("YOLO train wrapper (accepts extra args safely)")

    # 基本
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)   # .yaml (scratch) 或 .pt (接續訓練)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="runs/pig")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--device", type=str, default=None)

    # 增強/訓練細節（全部可選）
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.7)
    p.add_argument("--hsv_v", type=float, default=0.4)

    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=0.0)

    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)

    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    p.add_argument("--freeze", type=int, default=0)  # 從零訓練務必 0
    p.add_argument("--patience", type=int, default=20)

    # 你要求新增／常見但舊 wrapper 沒接到的
    p.add_argument("--erasing", type=float, default=None)           # 例：0.0 關閉
    p.add_argument("--auto_augment", type=str, default=None)        # 例："none" 或 "randaugment"
    p.add_argument("--rect", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--save_period", type=int, default=-1)           # -1 表示不額外保存
    p.add_argument("--close_mosaic", type=int, default=None)        # 例：10/20/25
    p.add_argument("--warmup_epochs", type=float, default=None)

    p.add_argument("--backbone_weights", type=str, default="",
                    help="只載入 backbone 的預訓練權重（分類/SSL），不載入偵測 head。")
    p.add_argument("--backbone_freeze", type=int, default=0,
                    help="載入 backbone 後先凍結前 N 層（等同於 --freeze），0 表示不凍結。")
    p.add_argument("--copy_paste", type=float, default=None)          # 例：0.10
    p.add_argument("--multi_scale", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--box", type=float, default=None)                  # box loss weight
    p.add_argument("--dfl", type=float, default=None)                  # dist focal loss weight
    p.add_argument("--cls", type=float, default=None) 
    
    # 顯示/驗證
    p.add_argument("--no_val", action="store_true")
    p.add_argument("--no_plots", action="store_true")

    return p.parse_args()

def load_backbone_only(det_model: nn.Module, ckpt_id: str):
    """
    只從分類/SSL ckpt 載入 'backbone' 權重到 Ultralytics 的 DetectionModel。
    - ckpt_id 可以是本機路徑或資產名（例：'yolo11m-cls.pt'）；會自動下載到快取
    - 嚴格限制到 backbone：YOLOv11 的 backbone 以 SPPF 結束，對應層索引 0..9
      （層 10 是 C2PSA，屬於 neck，不載入）
    """
    # 會自動處理本機/遠端資產；若不在就下載
    ckpt_path = check_file(ckpt_id)
    print(f"[Backbone] using weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) 取出 state_dict（容錯多種格式）
    if isinstance(ckpt, dict):
        m = ckpt.get("model", None)
        if isinstance(m, nn.Module):
            sd = m.state_dict()
        elif isinstance(m, dict) and "state_dict" in m:
            sd = m["state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    elif isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    # 2) 去掉常見前綴（如 'module.'）
    sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}

    det_sd = det_model.state_dict()

    # 3) 只允許 0..9（SPPF=9）這些層的鍵
    is_backbone = re.compile(r"^model\.(?:[0-9])\.")  # 0..9
    transfer = {}
    for k, v in sd.items():
        # 分類權重多半也是 'model.<idx>...' 的命名
        if k in det_sd and det_sd[k].shape == v.shape and is_backbone.match(k):
            transfer[k] = v

    det_model.load_state_dict(transfer, strict=False)
    print(f"[Backbone] 載入 {len(transfer)} 個 backbone 參數（層 0..9）；neck/head 全部隨機初始化。")

def guess_cls_from_yaml(model_path: str):
    p = Path(model_path)
    if p.suffix in {".yaml", ".yml"} and p.stem.startswith("yolo11"):
        return f"{p.stem}-cls.pt"  # 例：yolo11m.yaml -> yolo11m-cls.pt
    return ""

def resolve_cls_weight(name: str) -> str:
    # 先嘗試 Ultralytics 內建的檢查（本機或已知資產）
    try:
        return check_file(name)
    except Exception:
        pass
    # 不行就從官方 assets 下載到 weights/ 下面
    bases = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0",
        "https://github.com/ultralytics/assets/releases/download/v8.3.1",
    ]
    os.makedirs("weights", exist_ok=True)
    last_err = None
    for b in bases:
        url = f"{b}/{name}"
        dst = f"weights/{name}"
        try:
            print(f"[Backbone] downloading {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
            return dst
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"{name} not found; last error: {last_err}")



def main():
    args = parse_args()

    # 建模
    model = YOLO(args.model)  # .yaml -> scratch; .pt -> 續訓/微調

    # 組 overrides（只塞有值的）
    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "freeze": args.freeze,
        "patience": args.patience,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "rect": args.rect,
        "val": not args.no_val,
        "plots": not args.no_plots,
    }

    # 只有在你有提供的時候才加進去（避免某些 Ultralytics 版本不吃 None）
    if args.erasing is not None:
        overrides["erasing"] = args.erasing
    if args.auto_augment is not None:
        overrides["auto_augment"] = args.auto_augment
    if args.save_period is not None and args.save_period >= 0:
        overrides["save_period"] = args.save_period
    if args.close_mosaic is not None:
        overrides["close_mosaic"] = args.close_mosaic
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.model.endswith(".yaml"):
        overrides["pretrained"] = False
    # 只有在你有給值時才塞進去（避免 None 造成舊版不識別）
    if args.copy_paste is not None:
        overrides["copy_paste"] = args.copy_paste
    if args.multi_scale:
        overrides["multi_scale"] = True
    if args.box is not None:
        overrides["box"] = args.box
    if args.dfl is not None:
        overrides["dfl"] = args.dfl
    if args.cls is not None:
        overrides["cls"] = args.cls


    #  # 防違規（不允許偵測 .pt 當起點）
    # if args.model.endswith(".pt") and "-cls" not in Path(args.model).name:
    #     raise ValueError("作業規範：--model 需為 .yaml（scratch）或分類/SSL ckpt（-cls）。偵測 .pt 會違規。")

    # 自動猜分類權重（若沒給）
    bw = args.backbone_weights or guess_cls_from_yaml(args.model)  # yolo11m.yaml -> yolo11m-cls.pt
    if bw:
        try:
            bw_path = resolve_cls_weight(bw)
            load_backbone_only(model.model, bw_path)
            if args.backbone_freeze > 0:
                overrides["freeze"] = args.backbone_freeze
        except Exception as e:
            print(f"[Backbone] 載入失敗（改用隨機初始化）: {e}")


    # 開練
    model.train(**overrides)


if __name__ == "__main__":
    main()
