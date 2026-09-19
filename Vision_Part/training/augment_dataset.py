
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")
VARIANTS_PER_IMAGE = 2                               

def load_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((class_id, cx, cy, w, h))
    return boxes

def save_yolo_labels(label_path: Path, boxes):
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")

def horizontal_flip(img, boxes):
    flipped = cv2.flip(img, 1)
    new_boxes = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]
    return flipped, new_boxes

def brightness_contrast_jitter(img, boxes):
    alpha = random.uniform(0.75, 1.3)             
    beta = random.uniform(-25, 25)                  
    jittered = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return jittered, boxes                      

def slight_rotation(img, boxes, max_angle=8):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

    new_boxes = []
    for c, cx, cy, bw, bh in boxes:
        px, py = cx * w, cy * h
        corners = np.array([
            [px - bw * w / 2, py - bh * h / 2],
            [px + bw * w / 2, py - bh * h / 2],
            [px + bw * w / 2, py + bh * h / 2],
            [px - bw * w / 2, py + bh * h / 2],
        ])
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated_corners = corners_h @ rot_mat.T

        x_min, y_min = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
        x_max, y_max = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()

        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        if x_max <= x_min or y_max <= y_min:
            continue

        new_cx = ((x_min + x_max) / 2) / w
        new_cy = ((y_min + y_max) / 2) / h
        new_bw = (x_max - x_min) / w
        new_bh = (y_max - y_min) / h
        new_boxes.append((c, new_cx, new_cy, new_bw, new_bh))

    return rotated, new_boxes

def slight_scale(img, boxes, scale_range=(0.9, 1.15)):
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    if scale >= 1.0:
        x_off = (new_w - w) // 2
        y_off = (new_h - h) // 2
        out_img = resized[y_off:y_off + h, x_off:x_off + w]
        new_boxes = []
        for c, cx, cy, bw, bh in boxes:
            px, py = cx * new_w - x_off, cy * new_h - y_off
            pw, ph = bw * new_w, bh * new_h
            if px + pw / 2 <= 0 or px - pw / 2 >= w or py + ph / 2 <= 0 or py - ph / 2 >= h:
                continue
            new_boxes.append((c, px / w, py / h, pw / w, ph / h))
    else:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        out_img = cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h,
                                      pad_w, w - new_w - pad_w, cv2.BORDER_REPLICATE)
        new_boxes = []
        for c, cx, cy, bw, bh in boxes:
            px, py = cx * new_w + pad_w, cy * new_h + pad_h
            pw, ph = bw * new_w, bh * new_h
            new_boxes.append((c, px / w, py / h, pw / w, ph / h))

    return out_img, new_boxes

def gaussian_noise(img, boxes, sigma_range=(3, 10)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, boxes

ALL_TRANSFORMS = [
    ("flip", horizontal_flip),
    ("brightness", brightness_contrast_jitter),
    ("rotate", slight_rotation),
    ("scale", slight_scale),
    ("noise", gaussian_noise),
]

def make_augmented_variant(img, boxes):
    chosen = random.sample(ALL_TRANSFORMS, k=random.randint(2, 3))
    out_img, out_boxes = img.copy(), list(boxes)
    for _, fn in chosen:
        out_img, out_boxes = fn(out_img, out_boxes)
    return out_img, out_boxes

def main():
    ap = argparse.ArgumentParser(description="Triple a small YOLO dataset via augmentation.")
    ap.add_argument("--input", required=True, help="Folder with source images + matching .txt labels.")
    ap.add_argument("--output", required=True, help="Output folder for the full augmented dataset.")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Found {len(image_paths)} source images in {input_dir}")
    print(f"Generating {VARIANTS_PER_IMAGE} augmented variant(s) per image "
          f"(target total: ~{len(image_paths) * (VARIANTS_PER_IMAGE + 1)} images)")

    written = 0
    for i, img_path in enumerate(image_paths, 1):
        if i % 100 == 0:
            print(f"  ...{i}/{len(image_paths)} source images processed, {written} total images written")

        label_path = img_path.with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read: {img_path.name}, skipping.")
            continue
        boxes = load_yolo_labels(label_path)

        cv2.imwrite(str(output_dir / img_path.name), img)
        save_yolo_labels(output_dir / img_path.with_suffix(".txt").name, boxes)
        written += 1

        for v in range(VARIANTS_PER_IMAGE):
            aug_img, aug_boxes = make_augmented_variant(img, boxes)
            out_stem = f"{img_path.stem}_aug{v}"
            cv2.imwrite(str(output_dir / f"{out_stem}.jpg"), aug_img)
            save_yolo_labels(output_dir / f"{out_stem}.txt", aug_boxes)
            written += 1

    print(f"\nDone. {len(image_paths)} source images -> {written} total images in {output_dir}")

if __name__ == "__main__":
    main()
