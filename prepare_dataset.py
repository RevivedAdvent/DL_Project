import os, random, shutil
from pathlib import Path
from PIL import Image

RAW = Path("data/raw")
PROC = Path("data/processed")
random.seed(42)

def list_images(folder):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.open(src).convert("RGB")
        shutil.copy2(src, dst)
    except Exception:
        pass

def split(paths, train=0.7, val=0.15):
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * train)
    n_val = int(n * val)
    return paths[:n_train], paths[n_train:n_train+n_val], paths[n_train+n_val:]

if __name__ == "__main__":
    classes = ["hot","cold"]
    splits = {"train":{}, "val":{}, "test":{}}
    for cls in classes:
        imgs = list_images(RAW/cls)
        tr, va, te = split(imgs)
        splits["train"][cls] = tr
        splits["val"][cls] = va
        splits["test"][cls] = te
    for split_name in ["train","val","test"]:
        for cls, paths in splits[split_name].items():
            for i, src in enumerate(paths):
                dst = PROC / split_name / cls / f"{cls}_{i:05d}{src.suffix.lower()}"
                safe_copy(src, dst)
    print("Dataset prepared under data/processed/")
