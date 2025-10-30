#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from ultralytics import YOLO

def _coerce(v: str):
    s = str(v)
    # 1) 先數字
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?\d*\.\d+", s):
            return float(s)
    except Exception:
        pass
    # 2) 再 None
    if s.lower() == "none":
        return None
    # 3) 再 bool（只接受文字 true/false，不把 '0'/'1' 當 bool）
    if s.lower() in ("true", "yes", "y"):
        return True
    if s.lower() in ("false", "no", "n"):
        return False
    # 4) 逗號清單（一般參數會拆成 list；device 會在後面特別處理回字串）
    if "," in s:
        parts = [p for p in (x.strip() for x in s.split(",")) if p != ""]
        return [_coerce(p) for p in parts]
    return s

def _parse_unknown_to_kwargs(unknown):
    out = {}
    i = 0
    n = len(unknown)
    while i < n:
        tok = unknown[i]
        i += 1
        if not isinstance(tok, str) or not tok.startswith("--"):
            continue
        key = tok.lstrip("-")
        val = True  # 預設旗標
        if "=" in key:                 # --key=value
            k, v = key.split("=", 1)
            key, val = k, _coerce(v)
        elif i < n and not str(unknown[i]).startswith("--"):  # --key value
            val = _coerce(unknown[i])
            i += 1
        # dash->underscore
        key = key.replace("-", "_")
        out[key] = val
    return out

def _normalize_overrides(od: dict):
    # Ultralytics 的 device 建議字串：'cpu' 或 '0' 或 '0,1'
    if "device" in od:
        v = od["device"]
        if v is None:
            od.pop("device")
        elif isinstance(v, (list, tuple)):
            od["device"] = ",".join(str(x) for x in v)
        else:
            od["device"] = str(v)
    return od

def _enable_focal_loss(gamma: float = 1.5, alpha: float = 0.25):
    # 讓 YOLO 偵測的分類 loss 用 FocalLoss
    from ultralytics.utils.loss import FocalLoss
    try:
        # 新版路徑
        from ultralytics.utils.loss import v8DetectionLoss
    except Exception:
        # 舊版有時在這
        from ultralytics.models.yolo.detect.loss import v8DetectionLoss  # type: ignore

    old_init = v8DetectionLoss.__init__

    def _patched_init(self, *a, **kw):
        old_init(self, *a, **kw)
        # YOLO v8 的分類 loss 通常掛在 self.BCEcls
        import torch.nn as nn
        if hasattr(self, "BCEcls"):
            # 直接用官方的 FocalLoss（裡面已含 BCEWithLogits）
            self.BCEcls = FocalLoss(gamma=gamma, alpha=alpha)
    v8DetectionLoss.__init__ = _patched_init
    print(f"[patch] FocalLoss enabled for detection head: gamma={gamma}, alpha={alpha}")


def main():
    parser = argparse.ArgumentParser("Thin wrapper for Ultralytics YOLO: pass-through native args")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args, unknown = parser.parse_known_args()

    overrides = _parse_unknown_to_kwargs(list(unknown))
    overrides = _normalize_overrides(overrides)
    # 讓 --fl_gamma/--fl_alpha 可用，但不傳給 YOLO（避免報 SyntaxError）
    gamma = None
    if "fl_gamma" in overrides:
        gamma = float(overrides.pop("fl_gamma"))
    alpha = float(overrides.pop("fl_alpha", 0.25)) if "fl_alpha" in overrides else 0.25
    if gamma is not None:
        _enable_focal_loss(gamma=gamma, alpha=alpha)


    model = YOLO(args.model)
    print("[train] using overrides:")
    for k, v in overrides.items():
        print(f"  - {k}: {v}")
    model.train(data=args.data, **overrides)

if __name__ == "__main__":
    main()
