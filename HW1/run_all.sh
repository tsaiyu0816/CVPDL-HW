#!/usr/bin/env bash
set -euo pipefail

# 只保留安全的分割大小，避免 allocator 碎片化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:96"

# 0) 下載分類 backbone（若缺）
mkdir -p weights
if [ ! -f "weights/yolo11x-cls.pt" ]; then
  echo "[prep] downloading weights/yolo11x-cls.pt ..."
  URL1="https://github.com/ultralytics/assets/releases/download/v8.3.1/yolo11x-cls.pt"
  URL2="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 -o weights/yolo11x-cls.pt "$URL1" || curl -L --retry 3 -o weights/yolo11x-cls.pt "$URL2"
  else
    wget -O weights/yolo11x-cls.pt "$URL1" || wget -O weights/yolo11x-cls.pt "$URL2"
  fi
fi
[ -f "weights/yolo11x-cls.pt" ] || { echo "[ERR] 無法取得 yolo11x-cls.pt"; exit 1; }

# 1) 真的把資料夾改名成 images（不要用 symlink）
[ -L data/train/images ] && rm -f data/train/images
[ -d data/train/img ]  && mv data/train/img data/train/images
[ -L data/test/images ] && rm -f data/test/images
[ -d data/test/img ]   && mv data/test/img data/test/images

# 2) 重生 pig.yaml（避免 data/data/...）
python augment_offline.py || true
cat > data/pig.yaml <<'YAML'
path: data
train: [train/images, train_aug/images]
val: train/images
test: test/images
nc: 1
names: [pig]
YAML

# 3) 重新產生 YOLO labels（用 images/ 路徑）
python dataload.py --root data --train_img data/train/images --gt data/train/gt.txt --use_all_for_train

# 4) 清 cache（一定要）
rm -f data/train/*.cache data/test/*.cache data/train_aug/*.cache || true

# 5) 快速 sanity check
echo "[check] 有幾個標註檔？"; ls data/train/labels/*.txt 2>/dev/null | wc -l || true
echo "[check] pig.yaml："; head -5 data/pig.yaml

MODEL="yolo11x.yaml"
RUN_BASE="y11x_cls_backbone_v1"
SEEDS=("3407" "2025" "17")
WEIGHTS=()

# ===== Multi-Seed 三階段訓練（16GB 安全版，無 accumulate）=====
for SEED in "${SEEDS[@]}"; do
  RUN="${RUN_BASE}_s${SEED}"

  # Stage 1
  python train.py \
    --data data/pig.yaml --model "$MODEL" \
    --epochs 60 --batch 6 --imgsz 704 \
    --optimizer AdamW --lr0 0.0013 --weight_decay 0.0006 \
    --fliplr 0.70 --degrees 2.0 --translate 0.08 --scale 0.35 --shear 2.0 \
    --mosaic 0.25 --mixup 0.05 --copy_paste 0.18 \
    --hsv_v 0.55 --erasing 0.08 --auto_augment randaugment \
    --warmup_epochs 10 --close_mosaic 20 \
    --box 9.0 --dfl 2.0 --cls 0.15 \
    --name "$RUN" \
    --backbone_weights weights/yolo11x-cls.pt --freeze_backbone true \
    --seed "$SEED"

  # Stage 2
  RUN2="${RUN}_tight_s2"
  python train.py \
    --data data/pig.yaml \
    --model "runs/pig/${RUN}/weights/best.pt" \
    --epochs 100 --batch 3 --imgsz 896 \
    --optimizer SGD --lr0 0.001 --weight_decay 0.0006 \
    --fliplr 0.40 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
    --mosaic 0.0 --mixup 0.0 --hsv_v 0.35 --erasing 0.0 \
    --rect True --warmup_epochs 14 \
    --box 9.5 --dfl 2.0 --cls 0.12 \
    --name "$RUN2" --no_val \
    --freeze_backbone true \
    --seed "$SEED"

  # Stage 3
  RUN3="${RUN2}_hires_s3"
  python train.py \
    --data data/pig.yaml \
    --model "runs/pig/${RUN2}/weights/best.pt" \
    --epochs 30 --batch 1 --imgsz 1088 \
    --optimizer SGD --lr0 0.0006 --weight_decay 0.0006 \
    --fliplr 0.30 --degrees 0 --translate 0.02 --scale 0.12 --shear 0.0 \
    --mosaic 0.0 --mixup 0.0 --hsv_v 0.25 --erasing 0.0 \
    --rect True --warmup_epochs 6 \
    --box 9.5 --dfl 2.0 --cls 0.10 \
    --name "$RUN3" --no_val \
    --freeze_backbone true \
    --seed "$SEED"

  BEST="runs/pig/${RUN3}/weights/best.pt"
  LAST="runs/pig/${RUN3}/weights/last.pt"
  if   [ -f "$BEST" ]; then W="$BEST"
  elif [ -f "$LAST" ]; then W="$LAST"
  else
    echo "[WARN][SEED $SEED] 找不到最終權重，跳過此 seed。"; continue
  fi
  WEIGHTS+=("$W")
  echo "[SEED $SEED] final weight: $W"
done

# ===== 集成推論（WBF + HFlip + 多尺度）=====
if [ "${#WEIGHTS[@]}" -eq 0 ]; then
  echo "[ERR] 沒有任何可用權重，略過推論。"; exit 1
fi
WEIGHTS_CSV="$(IFS=, ; echo "${WEIGHTS[*]}")"


python inference.py \
  --weights "$WEIGHTS_CSV" \
  --test_dir data/test/images \
  --imgsz 1088 --conf 0.07 --final_conf 0.07 \
  --iou 0.55 \
  --wbf --wbf_hflip --wbf_scales 1.0\
  --wbf_iou 0.60 --wbf_conf_type avg \
  --topk 300 \
  --out_csv submission.csv

echo "[OK] submission.csv 已產生 ✅"
echo "用到的權重：$WEIGHTS_CSV"
