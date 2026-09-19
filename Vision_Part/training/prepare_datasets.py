import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(r"C:\Users\asus\Downloads")
MASTER_OUT = BASE_DIR / "NTI_Project" / "Master_YOLO_Dataset"

DATASETS = [
    {
        "name": "Smart Football- Object Detection.v11-od_sgd_optimizer.yolov8",
                                                                                    
        "mapping": {
            0: 2,                
            1: 0,                        
            2: 1,                      
            3: 0                            
        }
    },
    {
        "name": "football_detection.v8-dataset10k.yolov8",
                                                                                         
        "mapping": {
            0: 2,                
            1: -1,                
            2: 0,                        
            3: 0,                    
            4: 1                       
        }
    }
                                            
]

def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

clean_dir(MASTER_OUT / "train" / "images")
clean_dir(MASTER_OUT / "train" / "labels")
clean_dir(MASTER_OUT / "valid" / "images")
clean_dir(MASTER_OUT / "valid" / "labels")

all_images = []

for ds in DATASETS:
    ds_path = BASE_DIR / ds["name"]
    mapping = ds["mapping"]
    
    for split in ["train", "valid", "test"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        
        if not img_dir.exists():
            continue
            
        for img_file in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                all_images.append({
                    "img_path": img_file,
                    "lbl_path": lbl_file,
                    "mapping": mapping
                })

print(f"Total available images collected: {len(all_images)}")

random.shuffle(all_images)

target_train = 5600
target_valid = 1400

selected_train = all_images[:target_train]
selected_valid = all_images[target_train:target_train + target_valid]

def process_and_copy(subset, split_name):
    print(f"Processing {split_name} ({len(subset)} images)...")
    for idx, item in enumerate(subset):
        if idx % 1000 == 0:
            print(f"  Copied {idx}/{len(subset)}")
            
        img_path = item["img_path"]
        lbl_path = item["lbl_path"]
        mapping = item["mapping"]
        
        new_img_name = f"{img_path.parent.parent.parent.name[:8]}_{img_path.name}"
        new_lbl_name = f"{img_path.parent.parent.parent.name[:8]}_{lbl_path.name}"
        
        out_img = MASTER_OUT / split_name / "images" / new_img_name
        out_lbl = MASTER_OUT / split_name / "labels" / new_lbl_name
        
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    old_class = int(parts[0])
                    new_class = mapping.get(old_class, -1)
                    if new_class != -1:               
                        new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
        
        if len(new_lines) > 0:
            shutil.copy(img_path, out_img)
            with open(out_lbl, "w") as f:
                f.writelines(new_lines)

process_and_copy(selected_train, "train")
process_and_copy(selected_valid, "valid")

yaml_content = f"""path: {MASTER_OUT.absolute()}
train: train/images
val: valid/images

nc: 3
names: ['player', 'referee', 'ball']
"""
with open(MASTER_OUT / "data.yaml", "w") as f:
    f.write(yaml_content)

print("\nDONE! Dataset preparation complete. Total: 7000 images.")
