#!/usr/bin/env python3
import argparse
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

    # 顯示/驗證
    p.add_argument("--no_val", action="store_true")
    p.add_argument("--no_plots", action="store_true")

    return p.parse_args()

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

    # 開練
    model.train(**overrides)

if __name__ == "__main__":
    main()
