#!/usr/bin/env bash
# run.sh — one-call dataload (val split + valsync + chips + yaml), then native YOLO train + inference
# set -euo pipefail
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# ------------------------------
# 0) 清 cache / 舊 chips
# ------------------------------
rm -f  data/hw2_yolo/labels/train.cache data/hw2_yolo/labels/val.cache || true
rm -rf data/hw2_yolo/images/train_chips data/hw2_yolo/labels/train_chips || true
rm -rf data/hw2_yolo/images/train_cp data/hw2_yolo/labels/train_cp|| true
mkdir -p data/hw2_yolo/images data/hw2_yolo/labels

# ------------------------------
# 1) 一次到位：轉檔 → 切 val(10%) → ValSync → 依「目前 train」生 chips → 寫 YAML
#    * chips_per_img 在這裡就會生效 *
# ------------------------------
python dataload.py \
  --src "data/hw2/CVPDL_hw2/CVPDL_hw2" \
  --dst "data/hw2_yolo" \
  --yaml "data/hw2_fold0.yaml" \
  --names "car,hov,person,motorcycle" \
  --workers 12 \
  --val_ratio 0.10 --val_seed 2025 --skip_val_if_exists 1 --valsync 1 \
  --make_chips --chip_size 1024 --chips_per_img 0 --chip_jitter 0.15 \
  --chip_min_area 4 --chip_format jpg --jpg_quality 90 --clear_old_chips 1


# ------------------------------
# 1.5) copy paste class 2, 3
# ------------------------------
# python copy_paste_23.py \
#   --root data/hw2_yolo \
#   --yaml data/hw2_fold0.yaml \
#   --classes "2,3" \
#   --per_img 8 \
#   --prefer_bg \
#   --donor_area_min 0.001 --donor_area_max 0.015 \
#   --place_area_max 0.001 \
#   --scale_min 0.995 --scale_max 1.005 \
#   --tighten 0.08 \
#   --road_only --road_ratio 0.65 --road_s 60 --road_v 60 --green_gap 18 \
#   --color_match \
#   --cp_only \
#   --workers 8

# # 再清快取
# rm -f data/hw2_yolo/labels/train.cache

# ------------------------------
# 2) 單階段訓練（原生 YOLO 參數；只把 imgsz 拉到 1792）
# ------------------------------
MODEL="models/yolo11s-p2.yaml"
RUN="yolo11s-p2"
SEED=2025

python train.py \
  --data data/hw2_fold0.yaml \
  --model "$MODEL" \
  --imgsz 1920 \
  --epochs 150 \
  --batch 1 \
  --cls 0.6 --fl_gamma 1.5\
  --project runs/hw2 --name "$RUN" \
  --device 0 --seed "$SEED" --workers 12

# ------------------------------
# 3) 推論（slice 同步拉到 1792）
# ------------------------------
FINAL="runs/hw2/${RUN}/weights/best.pt"; [ -f "$FINAL" ] || FINAL="runs/hw2/${RUN}/weights/last.pt"
CONF_STR="0:0.10,1:0.08,2:0.05,3:0.05"

python inference.py \
  --mode full \
  --weights "$FINAL" \
  --test_dir data/hw2_yolo/images/test \
  --imgsz 1920 \
  --conf 0.05 \
  --iou 0.70 \
  --per_class_conf "$CONF_STR" \
  --out_csv submission11sp2.csv \
  --save_details \
  --viz_dir out_vis --viz_every 15 --viz_limit 60

echo "[✓] Done. submission11sp2_4.csv 已產生。"
