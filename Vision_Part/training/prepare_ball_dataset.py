import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(r"C:\Users\asus\Downloads")
PROJECT_DIR = BASE_DIR / "NTI_Project"
OUT_DIR = PROJECT_DIR / "Ball_Master_Dataset"

DATASETS = [
    PROJECT_DIR / "Ball_Only_Dataset",
    BASE_DIR / "football-ball.v4i.yolov8",
    BASE_DIR / "football-ball-detection.v3i.yolov8",
    BASE_DIR / "football-ball-detection.v2-ball-dataset.yolov8"
]

def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

clean_dir(OUT_DIR / "train" / "images")
clean_dir(OUT_DIR / "train" / "labels")
clean_dir(OUT_DIR / "valid" / "images")
clean_dir(OUT_DIR / "valid" / "labels")

all_images = []

for ds_path in DATASETS:
    if not ds_path.exists():
        continue
    for split in ["train", "valid", "test"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        if not img_dir.exists():
            continue
            
        for img_file in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                all_images.append((img_file, lbl_file))

print(f"Total available ball images collected: {len(all_images)}")

random.shuffle(all_images)

target_train = 4000
target_valid = 1000

selected_train = all_images[:target_train]
selected_valid = all_images[target_train:target_train + target_valid]

def process_and_copy(subset, split_name):
    print(f"Processing {split_name} ({len(subset)} images)...")
    for idx, (img_path, lbl_path) in enumerate(subset):
        if idx % 1000 == 0:
            print(f"  Copied {idx}/{len(subset)}")
            
        prefix = img_path.parent.parent.parent.name[:10]
        new_img_name = f"{prefix}_{img_path.name}"
        new_lbl_name = f"{prefix}_{lbl_path.name}"
        
        out_img = OUT_DIR / split_name / "images" / new_img_name
        out_lbl = OUT_DIR / split_name / "labels" / new_lbl_name
        
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                                                                               
                    new_lines.append(f"0 {' '.join(parts[1:])}\n")
        
        if len(new_lines) > 0:
            shutil.copy(img_path, out_img)
            with open(out_lbl, "w") as f:
                f.writelines(new_lines)

process_and_copy(selected_train, "train")
process_and_copy(selected_valid, "valid")

yaml_content = f"""path: {OUT_DIR.absolute()}
train: train/images
val: valid/images

nc: 1
names: ['ball']
"""
with open(OUT_DIR / "data.yaml", "w") as f:
    f.write(yaml_content)

print(f"\nDONE! Ball Master Dataset preparation complete. Ready for training.")
