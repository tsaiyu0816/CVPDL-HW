#!/usr/bin/env bash
# run.sh — dataload(每個 seed 重切) → train(3 seeds) → ensemble inference (WBF)

# set -euo pipefail
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL="models/yolo11s-p2.yaml"
BASE_RUN="y11s-p2"
SEEDS=(1314 17 3407)

# 先清一次共有暫存
rm -f  data/hw2_yolo/labels/train.cache data/hw2_yolo/labels/val.cache || true
rm -rf data/hw2_yolo/images/train_chips data/hw2_yolo/labels/train_chips || true
rm -rf data/hw2_yolo/images/train_cp data/hw2_yolo/labels/train_cp || true
mkdir -p data/hw2_yolo/images data/hw2_yolo/labels

WEIGHTS_LIST=()

for SEED in "${SEEDS[@]}"; do
  echo
  echo "==================== SEED ${SEED} ===================="

  YAML="data/hw2_fold_s${SEED}.yaml"
  RUN="${BASE_RUN}-s${SEED}"

  # 1) dataload：用這個 SEED 重新切 train/val（覆蓋先前 split）
  #    * 記得 skip_val_if_exists 設 0 才會真的重切
  #    * 其餘參數保持和你原本一致
  rm -f  data/hw2_yolo/labels/train.cache data/hw2_yolo/labels/val.cache || true
  rm -rf data/hw2_yolo/images/train_chips data/hw2_yolo/labels/train_chips || true
  rm -rf data/hw2_yolo

  python dataload.py \
    --src "data/hw2/CVPDL_hw2/CVPDL_hw2" \
    --dst "data/hw2_yolo" \
    --yaml "$YAML" \
    --names "car,hov,person,motorcycle" \
    --workers 12 \
    --val_ratio 0.10 --val_seed "$SEED" --skip_val_if_exists 0 --valsync 1 \
    --make_chips --chip_size 1024 --chips_per_img 0 --chip_jitter 0.15 \
    --chip_min_area 4 --chip_format jpg --jpg_quality 90 --clear_old_chips 1

  # 1.5) 若要 copy-paste，這段再自行解註（會寫回同一個 $YAML）
  # python copy_paste_23.py \
  #   --root data/hw2_yolo \
  #   --yaml "$YAML" \
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
  # rm -f data/hw2_yolo/labels/train.cache

  # 2) 訓練（超參數同一套）
  python train.py \
    --data "$YAML" \
    --model "$MODEL" \
    --imgsz 1920 \
    --epochs 150 \
    --batch 1 \
    --cls 0.6 --fl_gamma 1.5 \
    --project runs/hw2 --name "$RUN" \
    --device 0 --seed "$SEED" --workers 12

  FINAL="runs/hw2/${RUN}/weights/best.pt"; [ -f "$FINAL" ] || FINAL="runs/hw2/${RUN}/weights/last.pt"
  echo "[train] model picked: $FINAL"
  WEIGHTS_LIST+=("$FINAL")
done

# 3) Ensemble 推論（多權重 + WBF）
IFS=, ; WEIGHTS_CSV="${WEIGHTS_LIST[*]}"; unset IFS
echo "[ens] ensemble weights: $WEIGHTS_CSV"

CONF_STR="0:0.01,1:0.01,2:0.01,3:0.01"

python inference.py \
  --mode full \
  --weights "$WEIGHTS_CSV"\
  --wbf --wbf_iou 0.7 \
  --test_dir data/hw2_yolo/images/test \
  --imgsz 1920 \
  --conf 0.01 \
  --iou 0.70 \
  --per_class_conf "$CONF_STR" \
  --out_csv submission11sp2_mutiseed.csv \
  --save_details \
  --viz_dir out_vis_ens --viz_every 15 --viz_limit 60

echo "[✓] Done. submission11sp2_ens3.csv 已產生。"
