#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 訓練封裝（強化：
1) 只載入分類/SSL 的 backbone 權重到偵測模型；
2) 確實凍結 backbone（層 0..9）為 requires_grad=False，並列印凍結摘要；
3) overrides 僅塞有效鍵，避免 None 造成舊版不識別。
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.checks import check_file
import urllib.request
import random, numpy as np

# ---------------------------
# 解析工具
# ---------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")

def set_global_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 更穩定但稍慢
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set seed = {seed}")


def parse_args():
    p = argparse.ArgumentParser("YOLO train wrapper (clean & explicit freeze/backbone control)")

    # 基本
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)   # .yaml (scratch) 或 .pt (續訓)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="runs/pig")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=20)

    # 增強
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
    p.add_argument("--erasing", type=float, default=None)
    p.add_argument("--auto_augment", type=str, default=None)
    p.add_argument("--copy_paste", type=float, default=None)
    p.add_argument("--multi_scale", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--rect", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--save_period", type=int, default=-1)
    p.add_argument("--close_mosaic", type=int, default=None)
    p.add_argument("--warmup_epochs", type=float, default=None)

    # 損失權重
    p.add_argument("--box", type=float, default=None)
    p.add_argument("--dfl", type=float, default=None)
    p.add_argument("--cls", type=float, default=None)

    # 凍結選項
    p.add_argument("--freeze", type=int, default=0,
                   help="交給 Ultralytics 的 freeze（凍前 N 層）。僅在未使用 --freeze_backbone 時生效。")
    p.add_argument("--backbone_weights", type=str, default="",
                   help="只載入 backbone 的預訓練權重（分類/SSL），不載入偵測 head。")
    p.add_argument("--backbone_freeze", type=int, default=0,
                   help="已廢弛；為相容保留。若 >0 視同 --freeze_backbone=True。")
    p.add_argument("--freeze_backbone", type=str2bool, default=None,
                   help="凍結 YOLO backbone（層 0..9）。預設：若提供 --backbone_weights 則自動 True。")

    # 顯示/驗證
    p.add_argument("--no_val", action="store_true")
    p.add_argument("--no_plots", action="store_true")
   
    p.add_argument("--seed", type=int, default=None, help="全流程隨機種子（Ultralytics + torch + numpy + python）")
    


    return p.parse_args()


# ---------------------------
# Backbone 權重載入
# ---------------------------
def guess_cls_from_yaml(model_path: str) -> str:
    p = Path(model_path)
    if p.suffix in {".yaml", ".yml"} and p.stem.startswith("yolo11"):
        return f"{p.stem}-cls.pt"  # 例：yolo11x.yaml -> yolo11x-cls.pt
    return ""


def resolve_cls_weight(name: str) -> str:
    try:
        return check_file(name)  # 本機或快取資產
    except Exception:
        pass
    bases = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.1",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0",
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


def load_backbone_only(det_model: nn.Module, ckpt_id: str) -> int:
    """
    僅將分類/SSL checkpoint 的 'model.0..9.*'（SPPF=9）權重轉入偵測模型（neck/head 不載入）。
    回傳成功載入的參數個數。
    """
    ckpt_path = check_file(ckpt_id)
    print(f"[Backbone] using weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 取得 state_dict
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

    # 去掉 'module.' 前綴
    sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}

    det_sd = det_model.state_dict()
    is_backbone = re.compile(r"^model\.(?:[0-9])\.")  # 0..9

    transfer = {}
    for k, v in sd.items():
        if k in det_sd and det_sd[k].shape == v.shape and is_backbone.match(k):
            transfer[k] = v

    missing, unexpected = det_model.load_state_dict(transfer, strict=False)
    print(f"[Backbone] loaded {len(transfer)} tensors into backbone (layers 0..9).")
    if missing:
        print(f"[Backbone] note: {len(missing)} missing keys in det model (expected for neck/head).")
    if unexpected:
        print(f"[Backbone] note: {len(unexpected)} unexpected keys from ckpt (ignored).")
    return len(transfer)


# ---------------------------
# 凍結工具
# ---------------------------
def freeze_layers_by_index(det_model: nn.Module, idx_from: int, idx_to: int) -> Tuple[int, int]:
    """
    將 det_model.model[idx_from:idx_to+1] 的所有參數 requires_grad=False。
    回傳：(凍結參數數量, 參數總數)
    """
    assert hasattr(det_model, "model"), "det_model must have attribute .model (Ultralytics DetectModel)"
    total, frozen = 0, 0
    for i, m in enumerate(det_model.model):
        for p in m.parameters(recurse=True):
            total += 1
            if idx_from <= i <= idx_to:
                if p.requires_grad:
                    p.requires_grad = False
                frozen += 1
    print(f"[Freeze] layers {idx_from}..{idx_to} frozen "
          f"({frozen}/{total} tensors; {frozen/total*100:.1f}% of model params tensors).")
    return frozen, total


def summarize_trainable(det_model: nn.Module):
    t_all = sum(p.numel() for p in det_model.parameters())
    t_free = sum(p.numel() for p in det_model.parameters() if not p.requires_grad)
    t_tr = t_all - t_free
    print(f"[Freeze] trainable params: {t_tr:,} | frozen: {t_free:,} | total: {t_all:,}")


# ---------------------------
# 主流程
# ---------------------------
def main():
    args = parse_args()
    set_global_seed(args.seed)
    # 建模
    model = YOLO(args.model)  # .yaml -> scratch; .pt -> 續訓/微調

    # 組 overrides（只塞有值）
    overrides: Dict[str, object] = {
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
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    


    # ---- Backbone 權重（若指定）----
    bw = args.backbone_weights or guess_cls_from_yaml(args.model)
    loaded_backbone = False
    if bw:
        try:
            bw_path = resolve_cls_weight(bw)
            n_loaded = load_backbone_only(model.model, bw_path)
            loaded_backbone = n_loaded > 0
        except Exception as e:
            print(f"[Backbone] 載入失敗（改用隨機初始化）: {e}")

    # ---- 凍結策略 ----
    # freeze_backbone 預設：若有提供 backbone_weights 則 True，否則 False
    use_freeze_backbone = args.freeze_backbone
    if use_freeze_backbone is None:
        use_freeze_backbone = bool(bw)  # 自動啟用
    if args.backbone_freeze and args.backbone_freeze > 0:
        # 舊旗標相容：視同啟用凍結 backbone
        use_freeze_backbone = True

    if use_freeze_backbone:
        print("[Freeze] applying explicit requires_grad=False on backbone (layers 0..9).")
        freeze_layers_by_index(model.model, 0, 9)
        summarize_trainable(model.model)
        # 確保不再把 Ultralytics 的 'freeze' 也打開避免混淆
        overrides["freeze"] = 0
    else:
        # 未凍結 backbone：若使用者給了 freeze>0，交給 Ultralytics 原生處理
        overrides["freeze"] = int(max(0, args.freeze))
        if overrides["freeze"] > 0:
            print(f"[Freeze] delegating to Ultralytics: freeze first {overrides['freeze']} layers.")

    # ---- 開訓 ----
    model.train(**overrides)


if __name__ == "__main__":
    main()
