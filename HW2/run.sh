#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:96"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------- 0) 清 chips / cache（train 與 val 都清）--------
rm -rf data/hw2_yolo/images/train_chips data/hw2_yolo/labels/train_chips || true
rm -f  data/hw2_yolo/labels/train.cache data/hw2_yolo/labels/val.cache || true

# -------- 1) 整理資料：更大 chip、密集抽樣（僅 train 做 chips）--------
# 若你的 dataload.py 支援只處理 train 的選項，可在最後加上：--only_split train
python dataload.py \
  --src "data/hw2/CVPDL_hw2/CVPDL_hw2" \
  --dst "data/hw2_yolo" \
  --yaml "data/hw2.yaml" \
  --names "car,hov,person,motorcycle" \
  --make_chips --chip_size 1024 --chips_per_img 0 --chip_format jpg --jpg_quality 90 \
  --workers 12

# （可選）標註健檢：檔案存在才跑，不擋流程
if [ -f tools/check_yolo_labels.py ]; then
  python tools/check_yolo_labels.py data/hw2_yolo/labels/train_chips 4 || true
  python tools/check_yolo_labels.py data/hw2_yolo/labels/val 4 || true
fi
if [ -f tools/check_class_hist.py ]; then
  python tools/check_class_hist.py data/hw2_yolo/labels/train_chips || true
fi

MODEL="models/yolo11m.yaml"   # from scratch
SEED=3407
RUN="y11m_fs_ms_focal-1"

# -------- 2) Stage-1：Multi-Scale + Focal（召回↑，from-scratch 更好學）--------
# 若顯存吃緊，將 --imgsz 調回 1024 或把 --batch 用 auto 讓 UL 自動試算（我們這裡給 1024 穩健版）
python train.py \
  --data data/hw2.yaml --model "$MODEL" \
  --epochs 60 --batch 2 --imgsz 1536 \
  --optimizer AdamW --lr0 0.003 --lrf 0.01 --weight_decay 0.0005 \
  --warmup_epochs 8 --cos_lr True --patience 60 \
  --multi_scale True \
  --mosaic 0.50 --mixup 0.10 --copy_paste 0.30 --close_mosaic 12 \
  --degrees 0.0 --shear 0.0 --translate 0.05 --scale 0.50 --fliplr 0.50 \
  --hsv_h 0.010 --hsv_s 0.50 --hsv_v 0.30 \
  --box 7.5 --dfl 1.6 --cls 1.4  \
  --rect False --val True --plots True \
  --cache ram --workers 12 \
  --auto_augment None --erasing 0.0 \
  --project runs/hw2 --name "$RUN" --device 0 --seed "$SEED"

# -------- 3) Stage-2：高解析度精煉（定位↑，關大增強）--------
RUN2="${RUN}_s2"
S1W="runs/hw2/${RUN}/weights/best.pt"; [ -f "$S1W" ] || S1W="runs/hw2/${RUN}/weights/last.pt"
python train.py \
  --data data/hw2.yaml --model "$S1W" \
  --epochs 30 --batch 1 --imgsz 1792 \
  --optimizer AdamW --lr0 0.0009 --lrf 0.01 --weight_decay 0.0005 \
  --warmup_epochs 2 --cos_lr True \
  --mosaic 0.00 --mixup 0.00 --copy_paste 0.00 --close_mosaic 0 \
  --degrees 0.0 --shear 0.0 --translate 0.015 --scale 0.06 --fliplr 0.40 \
  --hsv_h 0.005 --hsv_s 0.35 --hsv_v 0.20 \
  --box 8.5 --dfl 1.9 --cls 1.4 \
  --rect True --val True --plots True \
  --auto_augment None --erasing 0.0 \
  --cache ram --workers 12 \
  --project runs/hw2 --name "$RUN2" --device 0 --seed "$SEED"

# -------- 4) 推論：可選自動校正 per-class conf（有工具檔才跑）--------
# 4) 推論：mAP50:95 取向（WBF）
FINAL="runs/hw2/${RUN2}/weights/best.pt"; [ -f "$FINAL" ] || FINAL="runs/hw2/${RUN2}/weights/last.pt"

# 保守 per-class 門檻：留住召回，交給 WBF 合併後再篩
CONF_STR="0:0.05,1:0.01,2:0.02,3:0.02"

python inference.py \
  --weights "$FINAL" \
  --test_dir data/hw2_yolo/images/test \
  --conf 0.01 \
  --slice 1280 --overlap 0.35 \
  --tta_hflip --tta_scales 1.0,1.2 \
  --wbf --wbf_iou 0.40 \
  --per_class_conf "$CONF_STR" \
  --out_csv submission_4.csv

