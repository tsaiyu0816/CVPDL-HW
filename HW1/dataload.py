# dataload.py
"""
把 data/train/gt.txt 轉成 YOLO 格式標註，並建立 train/val 檔案清單 + dataset yaml。
支援 gt.txt 行格式：
   Image_ID, x y w h cls [x y w h cls]...
或（不小心拿到預測格式也行）：
   Image_ID, score x y w h cls [score x y w h cls]...
"""
from pathlib import Path
import argparse, random
from typing import Union, Dict, List, Tuple
from utils import find_image_by_id, read_image_hw, clamp_box, tlwh_to_yolo, parse_boxes_tokens, set_seed
import re

def convert_gt_to_yolo(
    img_dir: Union[str, Path],
    gt_path: Union[str, Path],
    labels_dir: Union[str, Path],
    class_offset: int = 0
):
    img_dir    = Path(img_dir)
    gt_path    = Path(gt_path)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 先把每張圖的所有框收集起來
    boxes_per_img: Dict[Path, List[Tuple[float,float,float,float,int]]] = {}

    with gt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            # 允許用逗號/空白；第一個欄位是 image_id
            parts = [t for t in re.split(r"[,\s]+", ln) if t]
            image_id = parts[0].strip()
            if image_id.lower() in {"image_id", "imageid"}:
                continue
            tokens = parts[1:]

            # 你的格式：x,y,w,h（4 欄）
            if len(tokens) != 4:
                raise ValueError(f"Expect 4 numbers after Image_ID, got {len(tokens)}: {ln}")
            x, y, w, h = map(float, tokens)
            c = 0  # 單類：pig=0

            img_path = find_image_by_id(image_id, img_dir) or \
                       find_image_by_id(Path(image_id).stem, img_dir)
            if img_path is None:
                print(f"[WARN] image for id={image_id} not found under {img_dir}, skip.")
                continue

            boxes_per_img.setdefault(img_path, []).append((x, y, w, h, c))

    # 寫出 YOLO labels（同名 .txt），一次寫入所有框
    seen_imgs = set()
    for img_path, boxes in boxes_per_img.items():
        W, H = read_image_hw(img_path)
        label_path = labels_dir / (img_path.stem + ".txt")
        with label_path.open("w") as wf:
            for (x, y, w, h, c) in boxes:
                x, y, w, h = clamp_box(x, y, w, h, W, H)
                if w <= 0 or h <= 0:
                    continue
                cx, cy, nw, nh = tlwh_to_yolo(x, y, w, h, W, H)
                c = int(c) - class_offset if class_offset else int(c)
                wf.write(f"{c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        seen_imgs.add(img_path)

    # 為沒有標註的訓練圖建立空 label（允許負樣本）
    IMG_EXTS = [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]
    for p in img_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            lp = labels_dir / (p.stem + ".txt")
            if not lp.exists():
                lp.write_text("")

    return sorted(list(seen_imgs))

def split_train_val(img_dir: Union[str, Path], val_ratio=0.1, seed=42):
    from utils import natural_int
    set_seed(seed)
    img_paths = []
    for p in Path(img_dir).iterdir():
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]:
            img_paths.append(p)
    # 按數字自然序排序（保險）
    img_paths.sort(key=lambda p: (natural_int(p.stem) is None, natural_int(p.stem), p.stem))
    n = len(img_paths)
    v = max(1, int(n * val_ratio))
    random.shuffle(img_paths)
    val = img_paths[:v]
    train = img_paths[v:]
    return [str(p.resolve()) for p in train], [str(p.resolve()) for p in val]

def write_list_txt(paths, txt_path: Union[str, Path]):
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w") as f:
        for p in paths:
            f.write(p + "\n")

def write_dataset_yaml(yaml_path, train_txt, val_txt, nc=1, names=None):
    if names is None:
        names = ["pig"]
    train_p = Path(train_txt).resolve()
    val_p = Path(val_txt).resolve()
    content = (
        f"train: {train_p.as_posix()}\n"
        f"val: {val_p.as_posix()}\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yaml_path).write_text(content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", type=str)
    ap.add_argument("--train_img", default="data/train/images", type=str)  # 建議用 images/ 路徑（或用我先前的 symlink）
    ap.add_argument("--gt", default="data/train/gt.txt", type=str)
    ap.add_argument("--labels_dir", default="data/train/labels", type=str)
    ap.add_argument("--val_ratio", default=0.1, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--yaml_out", default="data/pig.yaml", type=str)
    ap.add_argument("--use_all_for_train", action="store_true",
                    help="不切驗證集；val.txt 會等於 train.txt")
    args = ap.parse_args()

    # 重新產生 labels
    convert_gt_to_yolo(args.train_img, args.gt, args.labels_dir)

    # 切分（或全部訓練）
    if args.use_all_for_train:
        all_imgs, _ = split_train_val(args.train_img, val_ratio=0.0, seed=args.seed)
        train_list = all_imgs
        val_list   = all_imgs  # YOLO 需要路徑存在，這樣最簡單
    else:
        train_list, val_list = split_train_val(args.train_img, args.val_ratio, args.seed)

    train_txt = Path(args.root) / "train.txt"
    val_txt   = Path(args.root) / "val.txt"
    write_list_txt(train_list, train_txt)
    write_list_txt(val_list,   val_txt)

    # 寫 yaml（用絕對路徑避免 data/data 問題）
    write_dataset_yaml(args.yaml_out, train_txt, val_txt, nc=1, names=["pig"])
    print(f"[OK] Wrote {args.yaml_out}\nTrain images: {len(train_list)}, Val images: {len(val_list)}")

if __name__ == "__main__":
    main()
