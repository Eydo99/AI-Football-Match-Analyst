
import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")
VAL_FRACTION = 0.15                                                                
SEED = 0
CLASS_NAMES = ["player", "referee", "ball"]

def find_matched_pairs(source: Path):
    pairs = []
    skipped = 0
    for img_path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS):
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            skipped += 1
    return pairs, skipped

def copy_matched_pairs(pairs, staging: Path):
    staging.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy2(img_path, staging / img_path.name)
        shutil.copy2(label_path, staging / label_path.name)

def run_augmentation(augment_script: Path, staging: Path, augmented: Path):
    subprocess.run(
        [sys.executable, str(augment_script), "--input", str(staging), "--output", str(augmented)],
        check=True,
    )

def base_stem(name: str) -> str:
                                                                                         
    if "_aug" in name:
        return name.rsplit("_aug", 1)[0]
    return name

def split_and_place(augmented: Path, final: Path):
    images_train, images_val = final / "images" / "train", final / "images" / "val"
    labels_train, labels_val = final / "labels" / "train", final / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in augmented.iterdir() if p.suffix.lower() in IMG_EXTS)

    groups = {}
    for img_path in img_paths:
        groups.setdefault(base_stem(img_path.stem), []).append(img_path)

    base_stems = sorted(groups.keys())
    random.Random(SEED).shuffle(base_stems)
    n_val_src = max(1, round(len(base_stems) * VAL_FRACTION))
    val_stems = set(base_stems[:n_val_src])

    n_train_imgs = n_val_imgs = 0
    for stem, imgs in groups.items():
        is_val = stem in val_stems
        img_dst = images_val if is_val else images_train
        lbl_dst = labels_val if is_val else labels_train
        for img_path in imgs:
            label_path = img_path.with_suffix(".txt")
            shutil.copy2(img_path, img_dst / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, lbl_dst / label_path.name)
            n_val_imgs += is_val
            n_train_imgs += not is_val

    return n_train_imgs, n_val_imgs, len(base_stems) - n_val_src, n_val_src

def write_data_yaml(final: Path) -> Path:
    yaml_path = final / "data.yaml"
    lines = [
        f"path: {final}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Folder with all extracted frames (mixed labeled/unlabeled).")
    ap.add_argument("--final", required=True, help="Output dataset root, e.g. yolov8_final_test")
    ap.add_argument("--augment-script", required=True, help="Path to augment_dataset.py")
    args = ap.parse_args()

    source = Path(args.source)
    final = Path(args.final)
    staging = final / "_staging_matched"
    augmented = final / "_staging_augmented"

    print(f"Scanning {source} for image+label pairs...")
    pairs, skipped = find_matched_pairs(source)
    print(f"Found {len(pairs)} matched pairs, skipped {skipped} images with no matching .txt")
    if not pairs:
        print("No matched pairs found -- nothing to do.")
        return

    print(f"Copying matched pairs to {staging}...")
    copy_matched_pairs(pairs, staging)

    print(f"Running augmentation via {args.augment_script}...")
    run_augmentation(Path(args.augment_script), staging, augmented)

    print(f"Splitting into train/val at {final} (grouped by source image, not by file)...")
    n_train_imgs, n_val_imgs, n_train_src, n_val_src = split_and_place(augmented, final)

    yaml_path = write_data_yaml(final)

    print("\nDone.")
    print(f"  Labeled source images : {len(pairs)}")
    print(f"  Train: {n_train_imgs} images (from {n_train_src} source photos)")
    print(f"  Val:   {n_val_imgs} images (from {n_val_src} source photos)")
    print(f"  data.yaml written to  : {yaml_path}")
    print(f"\n  Staging folders left at (safe to delete after you spot-check the result):")
    print(f"    {staging}")
    print(f"    {augmented}")

if __name__ == "__main__":
    main()
