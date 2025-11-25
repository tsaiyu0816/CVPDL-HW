# CVPDL HW3 — 使用 DDPM 生成 MNIST 手寫數字 (3×28×28 RGB)

本作業以 **DDPM (Denoising Diffusion Probabilistic Models)** 搭配輕量 **U-Net** 在 MNIST 上做非條件式影像生成。專案提供：
- 一鍵流程（訓練 → 生成 10k 圖片 → FID 評估）
- 生成過程可視化（`diffusion_grid.png`）

---

## 1. 資料下載與準備

此專案需要兩個東西：
1) 影像資料（最後要變成 `./mnist/` 底下很多張 `28×28 RGB .png`）
2) `mnist.npz`（可用於 FID 對照）

### 1.1 影像資料 → `./mnist/`
```bash
# 方式A：使用 gdown（建議）
pip install gdown

# 下載 Google Drive 檔案（影像壓縮檔）
gdown --id '1xVCJD6M6sE-tZJYxenLzvuEkSiYXig_F' -O mnist_images.zip
unzip -q mnist_images.zip -d mnist_raw

# 確認數量（期望約 60,000）
ls ./mnist | wc -l
```

若無法用 gdown，可手動下載：  
影像資料：https://drive.google.com/file/d/1xVCJD6M6sE-tZJYxenLzvuEkSiYXig_F/view  
解壓到 `mnist_raw/` 後，執行上述 Python 轉檔程式。

### 1.2 下載 `mnist.npz`
```bash
gdown --id '1QQMFWsdcCyD1HfCnwvIgrbcPPICDChCG' -O ./mnist.npz
# 或手動下載：https://drive.google.com/file/d/1QQMFWsdcCyD1HfCnwvIgrbcPPICDChCG/view
```

---

## 2. 安裝環境
```bash
pip install -r requirements.txt
# 需求重點：torch, torchvision, tqdm, pillow, pytorch-fid, pillow, numpy 等
```

---

## 3. 一鍵流程（train → generate → eval）

直接執行：
```bash
bash run.sh
```
---

## 4. 訓練（Train）

指令：
```bash
python train.py --epochs 100 --batch_size 128 --lr 2e-4 --timesteps 1000 --save ddpm_mnist.pt
```

參數說明：
- `--epochs`：訓練回合數（建議 50–100）
- `--batch_size`：訓練批次大小
- `--lr`：AdamW 的學習率
- `--timesteps`：擴散步數 T（常用 1000；也可試 500/200 做速度 vs 品質權衡）
- `--save`：模型權重輸出路徑

備註：本專案以**噪聲預測 MSE**作為 loss；隨機抽樣 timestep 訓練會讓 loss 在小區間內震盪屬正常。

---

## 5. 生成（Generate）

指令：
```bash
python generate.py --ckpt ddpm_mnist.pt --out_dir gen_imgs --num 10000 --batch_size 256 --make_grid --zip_name img_R13942126.zip
```

參數說明：
- `--ckpt`：訓練好的權重 `.pt`
- `--out_dir`：輸出資料夾（會生成 `00001.png` ~ `10000.png`）
- `--num`：生成張數（作業需求為 10k）
- `--batch_size`：推論批次（可視 GPU 放大到 512/1024）
- `--make_grid`：額外輸出 `diffusion_grid.png`（8×8 過程圖）
- `--zip_name`：打包檔名（ZIP 使用不壓縮模式以加速 I/O）

---

## 6. 評估（Evaluation, FID）

使用 `pytorch-fid` 比較生成影像與參考資料。

資料夾 vs 資料夾：
```bash
python -m pytorch_fid ./gen_imgs ./mnist
```

或與統計檔（npz）對比：
```bash
python -m pytorch_fid ./gen_imgs ./mnist.npz
```
---

## 7. 專案架構
```
.
├── train.py
├── generate.py
├── models.py
├── dataload.py
├── diffusion.py
├── requirements.txt
├── run.sh
├── mnist/           # 60k 張 28×28 RGB PNG（扁平）
├── mnist.npz        # 可選：FID 參考統計
└── gen_imgs/        # 生成結果（00001.png ~ 10000.png, diffusion_grid.png）
```


