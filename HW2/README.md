# CVPDL_HW2 — YOLO11s-P2 Small-Object Detection (Multi-Seed + WBF Ensemble)

End-to-end pipeline for **NTU CVPDL 2025 HW2**:  
**dataset download → data preprocessing (seeded split + ValSync + chips) → multi-seed training → ensemble inference (WBF) → `submission.csv`**.  
Default model: **YOLO11s-P2** (small-object-friendly) with **Focal Loss (γ=1.5, α=0.25)** enabled.

---

## 🔧 Environment Setup

```bash
# 1) Create & activate environment
conda create -n cvpdlhw2 python=3.10 -y
conda activate cvpdlhw2

# 2) Install dependencies (CUDA 12.1 wheels of PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

*Tip: On headless servers, prefer `opencv-python-headless` over `opencv-python`.*

---

## 📦 Dataset Download (Kaggle)

```bash
# Install & configure Kaggle CLI (place token at ~/.kaggle/kaggle.json)
pip install kaggle
chmod 600 ~/.kaggle/kaggle.json

# Download dataset to data/hw2
OUT=./data/hw2
mkdir -p "$OUT"
kaggle competitions download -c ntu-cvpdl-2025-hw-2 -p "$OUT" --force

# Unzip all archives
find "$OUT" -name "*.zip" -exec unzip -q -o {} -d "$OUT" \;
```

**Expected raw data root**
```
data/hw2/CVPDL_hw2/CVPDL_hw2
```

**Converted YOLO-format output**
```
data/hw2_yolo/
```

---

## ▶️ One-Click Full Pipeline

```bash
chmod +x run_final.sh
bash run_final.sh
```

**Outputs**
- Trained weights: `runs/hw2/<run_name>/weights/{best.pt|last.pt}`
- Submission CSV: `submission11sp2_mutiseed.csv`
- Visualizations: `out_vis_ens/`

---

## 📁 Directory Layout

```text
.
├─ data/
│  ├─ hw2/                      # Kaggle raw data (download + unzip here)
│  └─ hw2_yolo/                 # YOLO data produced by dataload.py (train/val/test + chips)
├─ models/
│  └─ yolo11s-p2.yaml           # Model definition (can be swapped)
├─ runs/
│  └─ hw2/<run_name>/           # Ultralytics training outputs
├─ out_vis_ens/                 # Ensemble inference visualizations
├─ dataload.py                  # Seeded split / ValSync / chips + YAML writer
├─ train.py                     # Training wrapper (YOLO11 + Focal Loss patch)
├─ inference.py                 # Single/multi-weight inference + WBF + submission.csv
├─ copy_paste_23.py             # (Optional) Copy-Paste augmentation utility
└─ run.sh                       # One-click orchestrator (multi-seed + ensemble)
```

---

## 🧠 Model Description — YOLO11s-P2 (Small-Object Friendly)

- **Why P2**: Standard YOLO heads output P3–P5 (strides 8/16/32). **YOLO11s-P2** adds a **P2 branch (stride 4)**, preserving fine spatial details and improving recall for small objects (e.g., pedestrian, motorcycle).  
- **Trade-offs**: Slightly higher memory/latency vs. base `s`, still real-time friendly.  
- **This repo**: P2 is paired with **high-res training (e.g., 1920)** and **chips** to further favor small targets. **Focal Loss (γ=1.5, α=0.25)** emphasizes hard/rare samples.

---

## 🧰 What `run_final.sh` Does

**Top-of-file knobs**
```bash
MODEL="models/yolo11s-p2.yaml"   # model YAML
BASE_RUN="2-y11s-p2"             # run name prefix
SEEDS=(1314 17 3407)             # multi-seed training
```

**Pipeline**
1. Cleanup caches & old chips.  
2. For each `SEED`:
   - **dataload**: re-split train/val (**ValSync on**), generate **chips**.
   - **train**: YOLO11s-P2 at high resolution (Focal Loss on).
   - Collect `best.pt` (fallback `last.pt`) for ensemble.
3. **Ensemble inference** with **WBF** (IoU=0.7), class-wise thresholds, CSV export, and optional visualization.

---

## 📜 Scripts & Tunable Arguments

### 1) `dataload.py` — Seeded Split / ValSync / Chips

**Purpose**
- Read Kaggle raw data.  
- Create train/val split with a given seed (`--val_seed`, `--val_ratio`).  
- **ValSync**: drop train labels overlapping with val.  
- Generate **chips** (tiling/cropping) for small objects.  
- Emit Ultralytics **YAML** (paths/classes).

**Usage**
```bash
python dataload.py \
  --src "data/hw2/CVPDL_hw2/CVPDL_hw2" \
  --dst "data/hw2_yolo" \
  --yaml "data/hw2_fold_s${SEED}.yaml" \
  --names "car,hov,person,motorcycle" \
  --workers 12 \
  --val_ratio 0.10 --val_seed "$SEED" --skip_val_if_exists 0 --valsync 1 \
  --make_chips --chip_size 1024 --chips_per_img 0 --chip_jitter 0.15 \
  --chip_min_area 4 --chip_format jpg --jpg_quality 90 --clear_old_chips 1
```

**(Optional) `copy_paste_23.py` — Copy-Paste augmentation**  
Uncomment in `run.sh` to enable. Typical knobs:
```text
--classes "2,3"                    # donor classes (e.g., person=2, motorcycle=3)
--per_img 8
--donor_area_min 0.001 --donor_area_max 0.015
--place_area_max 0.001
--scale_min 0.995 --scale_max 1.005
--tighten 0.08
--road_only --road_ratio 0.65 --road_s 60 --road_v 60 --green_gap 18
--color_match
--cp_only
```

---

### 2) `train.py` — Ultralytics YOLO + Focal Loss

**Purpose**
- Train **YOLO11s-P2** with high-res images and Focal Loss (γ=1.5, α=0.25).

**Usage**
```bash
python train.py \
  --data "data/hw2_fold_s${SEED}.yaml" \
  --model "models/yolo11s-p2.yaml" \
  --imgsz 1920 \
  --epochs 150 \
  --batch 1 \
  --cls 0.6 --fl_gamma 1.5 \
  --project runs/hw2 --name "${BASE_RUN}-s${SEED}" \
  --device 0 --seed ${SEED} --workers 12
```

---

### 3) `inference.py` — Single/Multi-weight Inference + WBF

**Purpose**
- Single or **comma-separated multi-weights** inference.  
- Fuse detections with **Weighted Boxes Fusion** (WBF).  
- Class-wise confidence thresholds.  
- Produce **Kaggle `submission.csv`** and optional visualizations.

**Usage**
```bash
python inference.py \
  --mode full \
  --weights "w1.pt,w2.pt,w3.pt" \
  --wbf --wbf_iou 0.7 \
  --test_dir data/hw2_yolo/images/test \
  --imgsz 1920 \
  --conf 0.01 \
  --iou 0.70 \
  --per_class_conf "0:0.01,1:0.01,2:0.01,3:0.01" \
  --out_csv submission11sp2_mutiseed.csv \
  --save_details \
  --viz_dir out_vis_ens --viz_every 15 --viz_limit 60
```

---

## ⚙️ Quick Knobs Reference

```text
run.sh
  MODEL, BASE_RUN, SEEDS

dataload.py
  --val_ratio, --val_seed, --skip_val_if_exists, --valsync
  --make_chips, --chip_size, --chips_per_img, --chip_jitter, --chip_min_area
  --names "car,hov,person,motorcycle"

copy_paste_23.py (optional)
  --classes, --per_img, --road_only, --color_match, ...

train.py
  --imgsz, --epochs, --batch, --cls, --fl_gamma, --device, --seed

inference.py
  --weights, --wbf, --wbf_iou, --conf, --iou, --per_class_conf, --viz_*
```

---

## ❓ FAQ

- **Out-of-Memory (OOM)** → lower `--imgsz` (e.g., 1536/1280) or `--batch`.  
- **Kaggle CLI auth errors** → ensure `~/.kaggle/kaggle.json` exists and `chmod 600` is set.  
- **Class order mismatches** → keep `--names "car,hov,person,motorcycle"` consistent.  
- **Missing weights for ensemble** → script picks `best.pt`, else `last.pt`.

---

## 📝 Notes

- Tuned for **small objects** (YOLO11s-P2 + high-res + Focal Loss).  
- For higher accuracy: try `yolo11m-p2.yaml` and/or enable Copy-Paste augmentation.  
- For faster inference: reduce `--imgsz` or switch to `yolo11s.yaml`.
