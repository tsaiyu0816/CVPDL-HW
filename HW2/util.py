#!/usr/bin/env python3
import random, numpy as np, torch, yaml
from pathlib import Path
from typing import Optional, List


def set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] {seed}")

def write_data_yaml(yaml_path: str, root: str, names: List[str],
                    train="images/train", val="images/train", test="images/test"):
    d = {
        "path": str(Path(root)),
        "train": train,
        "val": val,
        "test": test,
        "names": {i:n for i,n in enumerate(names)}
    }
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print(f"[ok] wrote data yaml -> {yaml_path}")


def check_label_ids(lbl_dir: str, num_classes: int):
    from pathlib import Path
    bad=[]
    for p in Path(lbl_dir).glob("*.txt"):
        for ln in p.read_text().splitlines():
            if not ln.strip(): continue
            try:
                cid=int(ln.split()[0])
                if cid<0 or cid>=num_classes:
                    bad.append((p.name,cid)); break
            except:
                bad.append((p.name,"parse")); break
    if bad:
        print(f"[WARN] label id out of range: {bad[:10]} ...")
    else:
        print("[check] label ids look OK.")
