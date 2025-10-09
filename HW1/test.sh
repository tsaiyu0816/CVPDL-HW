# # 1) 真的把資料夾改名成 images（不要用 symlink）
# [ -L data/train/images ] && rm -f data/train/images     # 刪掉舊 symlink（若存在）
# [ -d data/train/img ]  && mv data/train/img data/train/images

# [ -L data/test/images ] && rm -f data/test/images
# [ -d data/test/img ]   && mv data/test/img data/test/images

# # 2) 重生 pig.yaml（避免 path 拼成 data/data/...）
# python augment_offline.py

# # 讓 YOLO 同時吃兩份資料
# cat > data/pig.yaml <<'YAML'
# path: data
# train: [train/images, train_aug/images]
# val: train/images
# test: test/images
# nc: 1
# names: [pig]
# YAML

# # 3) 重新產生 YOLO labels（用 images/ 路徑）
# python dataload.py --root data --train_img data/train/images --gt data/train/gt.txt --use_all_for_train

# # 4) 清 cache（一定要）
# rm -f data/train/*.cache data/test/*.cache

# # 5) 快速 sanity check
# echo "[check] 有幾個標註檔？"
# ls data/train/labels/*.txt | wc -l
# echo "[check] pig.yaml："
# head -5 data/pig.yaml

# # MODEL="yolo11m.yaml"

# # RUN="y11m_stage1-1"
# # python train.py \
# #   --data data/pig.yaml --model "$MODEL" \
# #   --epochs 30 --batch 8 --imgsz 640 \
# #   --optimizer AdamW --lr0 0.0016 --weight_decay 0.0005 \
# #   --fliplr 0.7 --degrees 1.5 --translate 0.06 --scale 0.35 --shear 2.0 \
# #   --mosaic 0.25 --mixup 0.0 --hsv_v 0.45 \
# #   --erasing 0.1 --warmup_epochs 5 --close_mosaic 10 --auto_augment randaugment\
# #   --freeze 0 --name "$RUN" \
# #   --backbone_weights weights/yolo11m-cls.pt --backbone_freeze 5 \

# # RUN2="y11m_stage1-2"  
# # python train.py \
# #   --data data/pig.yaml --model runs/pig/${RUN}/weights/best.pt \
# #   --epochs 70 --batch 8 --imgsz 704 \
# #   --optimizer AdamW --lr0 0.0012 --weight_decay 0.0005 \
# #   --fliplr 0.7 --degrees 1.5 --translate 0.06 --scale 0.35 --shear 2.0 \
# #   --mosaic 0.25 --mixup 0.0 --hsv_v 0.45 \
# #   --erasing 0.1 --warmup_epochs 5 --close_mosaic 25 --auto_augment randaugment\
# #   --freeze 0 --name "$RUN2"


# # RUN3="${RUN2}_tight"

# # # 第二階段：專收斂定位（高 IoU）
# # python train.py \
# #   --data data/pig.yaml \
# #   --model runs/pig/${RUN2}/weights/best.pt \
# #   --epochs 60 --batch 4 --imgsz 960 \
# #   --optimizer SGD --lr0 0.0010 --weight_decay 0.0005 \
# #   --fliplr 0.4 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.3 --erasing 0.01\
# #   --rect True --patience 20 --save_period 20 \
# #   --freeze 0 --name "$RUN3"

# # W="runs/pig/${RUN3}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN3}/weights/last.pt"
# # python inference.py --weights "$W" --test_dir data/test/images --imgsz 960 --conf 0.07 --tta --out_csv submission.csv

# # MODEL="yolo11m.yaml"           # YOLOv11-M
# # RUN="y11m_v3_clsfree2"         # Stage-1 的 run 名

# # # ========== Stage 1（同你現在設定）==========
# # python train.py \
# #   --data data/pig.yaml --model "$MODEL" \
# #   --epochs 60 --batch 10 --imgsz 704 \
# #   --optimizer AdamW --lr0 0.0016 --weight_decay 0.0005 \
# #   --fliplr 0.7 --degrees 1.0 --translate 0.06 --scale 0.35 --shear 2.0 \
# #   --mosaic 0.20 --mixup 0.0 --hsv_v 0.50 \
# #   --erasing 0.03 --warmup_epochs 5 --close_mosaic 25 --auto_augment randaugment \
# #   --freeze 0 --name "$RUN" \
# #   --backbone_weights weights/yolo11m-cls.pt --backbone_freeze 2

# # # ========== Stage 2A（穩住：只動最末端，SGD，小步）==========
# # RUN2A="${RUN}_s2a_freeze8_stable"
# # python train.py \
# #   --data data/pig.yaml --model "runs/pig/${RUN}/weights/best.pt" \
# #   --epochs 30 --batch 12 --imgsz 832 \
# #   --optimizer SGD --lr0 0.0009 --weight_decay 0.0005 \
# #   --fliplr 0.55 --degrees 0.3 --translate 0.045 --scale 0.18 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.38 --erasing 0.0 \
# #   --warmup_epochs 10 --rect True \
# #   --freeze 8 --no_val --name "$RUN2A"

# # # ========== Stage 2B-1（極穩定：仍 freeze=8，LR 更低 + 長 warmup + 餘弦）==========
# # RUN2B1="${RUN}_s2b1_freeze8_cos"
# # python train.py \
# #   --data data/pig.yaml --model "runs/pig/${RUN2A}/weights/last.pt" \
# #   --epochs 24 --batch 12 --imgsz 832 \
# #   --optimizer AdamW --lr0 0.00040 --weight_decay 0.0006 \
# #   --fliplr 0.5 --degrees 0.2 --translate 0.04 --scale 0.16 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.36 --erasing 0.0 \
# #   --warmup_epochs 12 --rect True \
# #   --freeze 8 --no_val --name "$RUN2B1" \
# #   --lrf 0.01   #（你的 wrapper 已支援 lrf；cosine 在 8.3.x 會自動走 cos decay）

# # # ========== Stage 2B-2（輕解凍到 6：再細修）==========
# # RUN2B2="${RUN}_s2b2_freeze6_refine"
# # python train.py \
# #   --data data/pig.yaml --model "runs/pig/${RUN2B1}/weights/last.pt" \
# #   --epochs 30 --batch 12 --imgsz 832 \
# #   --optimizer AdamW --lr0 0.00035 --weight_decay 0.0006 \
# #   --fliplr 0.45 --degrees 0.2 --translate 0.04 --scale 0.16 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.36 --erasing 0.0 \
# #   --warmup_epochs 12 --rect True \
# #   --freeze 7 --no_val --name "$RUN2B2" \
# #   --lrf 0.01

# # # ========== Stage 3（tight：高 IoU 定位）==========
# # RUN3="${RUN2B2}_tight"
# # python train.py \
# #   --data data/pig.yaml --model "runs/pig/${RUN2B2}/weights/last.pt" \
# #   --epochs 40 --batch 6 --imgsz 1024 \
# #   --optimizer SGD --lr0 0.0007 --weight_decay 0.0005 \
# #   --fliplr 0.35 --degrees 0 --translate 0.03 --scale 0.16 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.30 --erasing 0.0 \
# #   --rect True --patience 25 --save_period 20 \
# #   --freeze 10 --no_val --name "$RUN3"

# # # ========== Inference ==========
# # W="runs/pig/${RUN3}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN3}/weights/last.pt"
# # python inference.py --weights "$W" --test_dir data/test/images --imgsz 1024 --conf 0.12 --tta --out_csv submission.csv


# # MODEL="yolo11m.yaml"
# # RUN="y11m_stage1_imcls_test_v4"   # 標記一下用了 ImageNet backbone

# # # ---------- Stage 1：沿用你的參數，只多載入 ImageNet backbone，輕度凍結 ----------
# # python train.py \
# #   --data data/pig.yaml --model "$MODEL" \
# #   --epochs 100 --batch 8 --imgsz 704 \
# #   --optimizer AdamW --lr0 0.0016 --weight_decay 0.0005 \
# #   --fliplr 0.8 --degrees 2 --translate 0.1 --scale 0.35 --shear 2.0 \
# #   --mosaic 0.3 --mixup 0.0 --hsv_v 0.5 \
# #   --erasing 0.1 --warmup_epochs 5 --close_mosaic 30 --auto_augment randaugment \
# #   --freeze 0 --name "$RUN" \
# #   --backbone_weights weights/yolo11m-cls.pt --backbone_freeze 3


# # RUN2="${RUN}_tight"

# # # ---------- Stage 2（tight）：沿用你的參數，只加長 warmup 降低震盪 ----------
# # python train.py \
# #   --data data/pig.yaml \
# #   --model runs/pig/${RUN}/weights/best.pt \
# #   --epochs 60 --batch 4 --imgsz 960 \
# #   --optimizer SGD --lr0 0.0010 --weight_decay 0.0005 \
# #   --fliplr 0.4 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
# #   --mosaic 0.0 --mixup 0.0 --hsv_v 0.3 --erasing 0.01 \
# #   --rect True --patience 20 \
# #   --freeze 0 --name "$RUN2"

# # # ---------- 推論（跟你一致） ----------
# # W="runs/pig/${RUN2}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN2}/weights/last.pt"
# # # python inference.py --weights "$W" --test_dir data/test/images --imgsz 960 --conf 0.07 --tta --out_csv submission.csv

# # python inference.py \
# #   --weights "$W" \
# #   --test_dir data/test/images \
# #   --imgsz 960 --conf 0.05 --final_conf 0.12 \
# #   --iou 0.65 \
# #   --wbf --wbf_hflip --wbf_scales 1.05,1.0,0.95 \
# #   --wbf_iou 0.60 --wbf_conf_type max \
# #   --topk 250 \
# #   --out_csv submission.csv

# MODEL="yolo11m.yaml"
# RUN="y11m_cls_backbone_v4"

# # ---------- Stage 1：用 ImageNet backbone，強化左右視角＆亮暗，多尺度+CopyPaste，定位權重拉高 ----------
# python train.py \
#   --data data/pig.yaml --model "$MODEL" \
#   --epochs 60 --batch 8 --imgsz 704 \
#   --optimizer AdamW --lr0 0.0015 --weight_decay 0.0005 \
#   --fliplr 0.85 --degrees 1.5 --translate 0.08 --scale 0.30 --shear 1.5 \
#   --mosaic 0.20 --mixup 0.0 --copy_paste 0.10 \
#   --hsv_v 0.55 --erasing 0.05 --auto_augment randaugment \
#   --warmup_epochs 8 --close_mosaic 15 --multi_scale True \
#   --box 8.5 --dfl 2.0 --cls 0.25 \
#   --freeze 10 --name "$RUN" \
#   --backbone_weights weights/yolo11m-cls.pt 

# # ---------- Stage 2（tight）：高 IoU 收斂，降低震盪（長 warmup、低 LR、可選局部凍結） ----------
# RUN2="${RUN}_tight_s2"
# python train.py \
#   --data data/pig.yaml \
#   --model runs/pig/${RUN}/weights/best.pt \
#   --epochs 100 --batch 6 --imgsz 960 \
#   --optimizer SGD --lr0 0.001 --weight_decay 0.0005 \
#   --fliplr 0.35 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
#   --mosaic 0.0 --mixup 0.0 --hsv_v 0.35 --erasing 0.0 \
#   --rect True --warmup_epochs 12 \
#   --box 9.0 --dfl 2.0 --cls 0.20 \
#   --freeze 10 --name "$RUN2"

# # ---------- 推論（WBF + HFlip，保守最終門檻） ----------
# W="runs/pig/${RUN2}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN2}/weights/last.pt"

# python inference.py \
#   --weights "$W" \
#   --test_dir data/test/images \
#   --imgsz 960 --conf 0.2 --final_conf 0.12 \
#   --iou 0.65 \
#   --wbf --wbf_hflip --wbf_scales 1.05,1.0,0.95 \
#   --wbf_iou 0.60 --wbf_conf_type avg \
#   --topk 250 \
#   --out_csv submission.csv

# echo "[OK] submission.csv 已產生 ✅"
# echo "用到的權重：$W"


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
    --epochs 1 --batch 6 --imgsz 704 \
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
    --epochs 1 --batch 3 --imgsz 896 \
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
    --epochs 1 --batch 1 --imgsz 1088 \
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
