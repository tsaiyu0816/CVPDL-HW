# CVPDL_HW1
### This project focuses on automated detection of pigs
## Environment Setup
```bash
pip install -r requirements.txt
```
## Train model and Inference
### Run preprocessing, training, and inference with a single command
```bash 
bash run_all_final.sh
```
## Training Recipe (3 Stages × Multi‑Seed)

Default seeds:
```bash
SEEDS=("3407" "2025" "17")
```

### Stage 1 — Warm‑up at 704px (frozen backbone, strong aug)
- `optimizer=AdamW`, `imgsz=704`, `batch=6`, epochs=60
- Stronger augmentation early (mosaic/mixup/copy_paste/auto_augment)
- **Backbone frozen** with classification weights (`yolo11x-cls.pt`)

<details>
<summary>Command</summary>

```bash
python train.py \
  --data data/pig.yaml --model yolo11x.yaml \
  --epochs 60 --batch 6 --imgsz 704 \
  --optimizer AdamW --lr0 0.0013 --weight_decay 0.0006 \
  --fliplr 0.70 --degrees 2.0 --translate 0.08 --scale 0.35 --shear 2.0 \
  --mosaic 0.25 --mixup 0.05 --copy_paste 0.18 \
  --hsv_v 0.55 --erasing 0.08 --auto_augment randaugment \
  --warmup_epochs 10 --close_mosaic 20 \
  --box 9.0 --dfl 2.0 --cls 0.15 \
  --backbone_weights weights/yolo11x-cls.pt --freeze_backbone true \
  --seed <SEED> --name <RUN>
```
</details>

### Stage 2 — Convergence at 896px (rect, clean aug)
- `optimizer=SGD`, `imgsz=896`, `batch=3`, epochs=100
- **Rect training**, **no mosaic/mixup**, milder color jitter
- `--no_val` for speed (toggle off if you want mAP curves)

<details>
<summary>Command</summary>

```bash
python train.py \
  --data data/pig.yaml \
  --model runs/pig/<RUN>/weights/best.pt \
  --epochs 100 --batch 3 --imgsz 896 \
  --optimizer SGD --lr0 0.001 --weight_decay 0.0006 \
  --fliplr 0.40 --degrees 0 --translate 0.03 --scale 0.18 --shear 0.0 \
  --mosaic 0.0 --mixup 0.0 --hsv_v 0.35 --erasing 0.0 \
  --rect True --warmup_epochs 14 \
  --box 9.5 --dfl 2.0 --cls 0.12 \
  --name <RUN>_tight_s2 --no_val \
  --freeze_backbone true \
  --seed <SEED>
```
</details>

### Stage 3 — Hi‑res at 1088px (batch=1 fine‑tune)
- `optimizer=SGD`, `imgsz=1088`, `batch=1`, epochs=30
- Keep rect training; no mosaic/mixup; low LR, short warm‑up

<details>
<summary>Command</summary>

```bash
python train.py \
  --data data/pig.yaml \
  --model runs/pig/<RUN>_tight_s2/weights/best.pt \
  --epochs 30 --batch 1 --imgsz 1088 \
  --optimizer SGD --lr0 0.0006 --weight_decay 0.0006 \
  --fliplr 0.30 --degrees 0 --translate 0.02 --scale 0.12 --shear 0.0 \
  --mosaic 0.0 --mixup 0.0 --hsv_v 0.25 --erasing 0.0 \
  --rect True --warmup_epochs 6 \
  --box 9.5 --dfl 2.0 --cls 0.10 \
  --name <RUN>_tight_s2_hires_s3 --no_val \
  --freeze_backbone true \
  --seed <SEED>
```
</details>

> **Tips**
> - If your dataset is larger, consider **unfreezing the backbone** in Stage 2/3 and reducing `lr0` accordingly.
> - For tighter VRAM, lower `--imgsz` or `--batch` per stage.

---

## Inference & Submission

The pipeline collects the final weights from each seed and runs **ensemble inference**:

```bash
python inference.py \
  --weights "<w1>,<w2>,<w3>" \
  --test_dir data/test/images \
  --imgsz 1088 --conf 0.07 --final_conf 0.07 \
  --iou 0.55 \
  --wbf --wbf_hflip --wbf_scales 1.0 \
  --wbf_iou 0.60 --wbf_conf_type avg \
  --topk 300 \
  --out_csv submission.csv
```

**Submission format** (`submission.csv`):
```
Image_ID,PredictionString
000123.jpg,0.91 12 34 56 78 0 0.88 100 50 40 60 0 ...
```
Where `PredictionString` is space‑separated chunks of `score x y w h cls` (pixel units, top‑left origin), and `cls=0` denotes **pig**.

---