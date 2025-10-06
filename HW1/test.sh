# 1) 真的把資料夾改名成 images（不要用 symlink）
[ -L data/train/images ] && rm -f data/train/images     # 刪掉舊 symlink（若存在）
[ -d data/train/img ]  && mv data/train/img data/train/images

[ -L data/test/images ] && rm -f data/test/images
[ -d data/test/img ]   && mv data/test/img data/test/images

# 2) 重生 pig.yaml（避免 path 拼成 data/data/...）
python augment_offline.py

# 讓 YOLO 同時吃兩份資料
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
rm -f data/train/*.cache data/test/*.cache

# 5) 快速 sanity check
echo "[check] 有幾個標註檔？"
ls data/train/labels/*.txt | wc -l
echo "[check] pig.yaml："
head -5 data/pig.yaml

# MODEL="yolo11m.yaml"


# RUN="y11m_stage1-1"
# python train.py \
#   --data data/pig.yaml --model "$MODEL" \
#   --epochs 30 --batch 8 --imgsz 640 \
#   --optimizer AdamW --lr0 0.0016 --weight_decay 0.0005 \
#   --fliplr 0.7 --degrees 1.5 --translate 0.06 --scale 0.35 --shear 2.0 \
#   --mosaic 0.25 --mixup 0.0 --hsv_v 0.45 \
#   --erasing 0.1 --warmup_epochs 5 --close_mosaic 10 --auto_augment randaugment\
#   --freeze 0 --name "$RUN" \
#   --backbone_weights weights/yolo11m-cls.pt --backbone_freeze 5 \

# RUN2="y11m_stage1-2"  
# python train.py \
#   --data data/pig.yaml --model runs/pig/${RUN}/weights/best.pt \
#   --epochs 70 --batch 8 --imgsz 704 \
#   --optimizer AdamW --lr0 0.0012 --weight_decay 0.0005 \
#   --fliplr 0.7 --degrees 1.5 --translate 0.06 --scale 0.35 --shear 2.0 \
#   --mosaic 0.25 --mixup 0.0 --hsv_v 0.45 \
#   --erasing 0.1 --warmup_epochs 5 --close_mosaic 25 --auto_augment randaugment\
#   --freeze 0 --name "$RUN2"


# RUN3="${RUN2}_tight"

# # 第二階段：專收斂定位（高 IoU）
# python train.py \
#   --data data/pig.yaml \
#   --model runs/pig/${RUN2}/weights/best.pt \
#   --epochs 60 --batch 4 --imgsz 960 \
#   --optimizer SGD --lr0 0.0010 --weight_decay 0.0005 \
#   --fliplr 0.4 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
#   --mosaic 0.0 --mixup 0.0 --hsv_v 0.3 --erasing 0.01\
#   --rect True --patience 20 --save_period 20 \
#   --freeze 0 --name "$RUN3"

# W="runs/pig/${RUN3}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN3}/weights/last.pt"
# python inference.py --weights "$W" --test_dir data/test/images --imgsz 960 --conf 0.07 --tta --out_csv submission.csv

MODEL="yolo11m.yaml"

# —— Stage 1：用 ImageNet backbone，增強收斂但不過猛，凍結層數降低 ——
RUN="y11m_s1_clsfree2"
python train.py \
  --data data/pig.yaml --model "$MODEL" \
  --epochs 60 --batch 10 --imgsz 704 \
  --optimizer AdamW --lr0 0.0016 --weight_decay 0.0005 \
  --fliplr 0.8 --degrees 1.0 --translate 0.06 --scale 0.35 --shear 1.0 \
  --mosaic 0.20 --mixup 0.0 --hsv_v 0.50 \
  --erasing 0.03 --warmup_epochs 5 --close_mosaic 10 --auto_augment randaugment \
  --freeze 0 --name "$RUN" \
  --backbone_weights weights/yolo11m-cls.pt --backbone_freeze 2

# —— Stage 2：接 Stage-1 的 best.pt，解凍、拉一點解析度，減少干擾型增強 ——
RUN2="${RUN}_s2_best"
python train.py \
  --data data/pig.yaml --model runs/pig/${RUN}/weights/best.pt \
  --epochs 80 --batch 8 --imgsz 832 \
  --optimizer AdamW --lr0 0.0009 --weight_decay 0.0005 \
  --fliplr 0.6 --degrees 0.5 --translate 0.05 --scale 0.25 --shear 0.0 \
  --mosaic 0.10 --mixup 0.0 --hsv_v 0.40 \
  --erasing 0.0 --close_mosaic 20 --rect True \
  --freeze 0 --name "$RUN2"

# —— Stage 3（tight）：專注定位，高 IoU、無馬賽克，長邊更高解析 ——
RUN3="${RUN2}_tight"
python train.py \
  --data data/pig.yaml \
  --model runs/pig/${RUN2}/weights/best.pt \
  --epochs 50 --batch 6 --imgsz 960 \
  --optimizer SGD --lr0 0.0008 --weight_decay 0.0005 \
  --fliplr 0.4 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
  --mosaic 0.0 --mixup 0.0 --hsv_v 0.30 --erasing 0.0 \
  --rect True --patience 30 --save_period 20 \
  --freeze 0 --name "$RUN3"

# 推論（把 conf 調回比較保守，避免太多誤檢）
W="runs/pig/${RUN3}/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN3}/weights/last.pt"
python inference.py --weights "$W" --test_dir data/test/images --imgsz 960 --conf 0.15 --tta --out_csv submission.csv
