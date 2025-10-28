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

def main():
    parser = argparse.ArgumentParser("Thin wrapper for Ultralytics YOLO: pass-through native args")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args, unknown = parser.parse_known_args()

    overrides = _parse_unknown_to_kwargs(list(unknown))
    overrides = _normalize_overrides(overrides)

    model = YOLO(args.model)
    print("[train] using overrides:")
    for k, v in overrides.items():
        print(f"  - {k}: {v}")
    model.train(data=args.data, **overrides)

if __name__ == "__main__":
    main()
