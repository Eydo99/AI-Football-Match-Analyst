
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GT_COLUMNS = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
              "conf", "unused1", "unused2", "unused3"]

SIGMA_SCALE = 3.0                                          
SIGMA_MIN = 2.5                                                           
                                                                        
SIGMA_MAX = 12.0                                                           
                                                                       
def parse_gameinfo(gameinfo_path: Path) -> dict:
    mapping = {}
    text = gameinfo_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = re.match(r"trackletID_(\d+)\s*=\s*([^\n;]+)", line.strip(), re.IGNORECASE)
        if not m:
            continue
        track_id = int(m.group(1))
        role_text = m.group(2).strip().lower()
        if "ball" in role_text:
            mapping[track_id] = "ball"
    return mapping

def sigma_from_bbox(bb_width, bb_height):
    diagonal = (bb_width ** 2 + bb_height ** 2) ** 0.5
    sigma = diagonal / SIGMA_SCALE
    return min(max(sigma, SIGMA_MIN), SIGMA_MAX)

def make_gaussian_heatmap(w, h, cx, cy, sigma):
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    return (heatmap * 255).astype(np.uint8)

def process_sequence(seq_dir: Path, heatmaps_dir: Path, rows: list, stride: int = 1):
    gt_path = seq_dir / "gt" / "gt.txt"
    gameinfo_path = seq_dir / "gameinfo.ini"
    img_dir = seq_dir / "img1"

    if not gt_path.exists() or not gameinfo_path.exists() or not img_dir.exists():
        print(f"  Skipping {seq_dir.name} -- missing gt.txt, gameinfo.ini, or img1/")
        return 0

    id_to_class = parse_gameinfo(gameinfo_path)
    ball_ids = {tid for tid, cls in id_to_class.items() if cls == "ball"}
    if not ball_ids:
        print(f"  Skipping {seq_dir.name} -- no ball track ID found in gameinfo.ini")
        return 0

    gt = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)
    ball_gt = gt[gt["track_id"].isin(ball_ids)].copy()
                                                                              
    ball_gt = ball_gt.sort_values("conf", ascending=False).drop_duplicates("frame", keep="first")
    ball_by_frame = {int(r.frame): r for r in ball_gt.itertuples()}

    frame_nums = sorted(int(p.stem) for p in img_dir.glob("*.jpg") if p.stem.isdigit())
    if not frame_nums:
        return 0
    first_frame = frame_nums[0]

    sample_img = cv2.imread(str(img_dir / f"{frame_nums[0]:06d}.jpg"))
    if sample_img is None:
        print(f"  Skipping {seq_dir.name} -- could not read a sample frame")
        return 0
    img_h, img_w = sample_img.shape[:2]

    written = 0
    eligible_count = 0                                                                  
                                                                                         
    frame_set = set(frame_nums)
    for n in frame_nums:
        n2, n1 = n - 1, n - 2                                                          
        if n1 < first_frame or n2 < first_frame:
            continue                                                          
        if n1 not in frame_set or n2 not in frame_set:
            continue                                                        

        ball_row = ball_by_frame.get(n)
        if ball_row is None:
            continue                                                                  

        eligible_count += 1
        if (eligible_count - 1) % stride != 0:
            continue                                                                

        cx = ball_row.bb_left + ball_row.bb_width / 2
        cy = ball_row.bb_top + ball_row.bb_height / 2
                                                                            
        cx = min(max(cx, 0), img_w - 1)
        cy = min(max(cy, 0), img_h - 1)

        sigma = sigma_from_bbox(ball_row.bb_width, ball_row.bb_height)
        heatmap = make_gaussian_heatmap(img_w, img_h, cx, cy, sigma)
        heatmap_name = f"{seq_dir.name}_{n:06d}.png"
        cv2.imwrite(str(heatmaps_dir / heatmap_name), heatmap)

        rows.append({
            "sequence": seq_dir.name,
            "frame_n2": str(img_dir / f"{n2:06d}.jpg"),
            "frame_n1": str(img_dir / f"{n1:06d}.jpg"),
            "frame_n": str(img_dir / f"{n:06d}.jpg"),
            "heatmap": str(heatmaps_dir / heatmap_name),
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "sigma": round(sigma, 2),
            "img_w": img_w,
            "img_h": img_h,
        })
        written += 1

    return written

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Path to .../tracking (contains train/SNMOT-XXX)")
    ap.add_argument("--output", required=True, help="Output folder for index.csv + heatmaps/")
    ap.add_argument("--stride", type=int, default=1,
                     help="Keep 1 out of every N usable (ball-labeled) triplets per sequence. "
                          "Default 1 = keep all (~42k total across 57 seqs). Use --stride 4 "
                          "to land around 10-14k total, matching the original TrackNet paper's "
                          "dataset scale while still touching every sequence for diversity.")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    heatmaps_dir = output / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p.parent.parent for p in source.rglob("gt/gt.txt"))
    if not seq_dirs:
        print(f"No sequences found under {source}.")
        return

    print(f"Found {len(seq_dirs)} sequences. Building triplets (stride={args.stride})...\n")

    rows = []
    for seq_dir in seq_dirs:
        n = process_sequence(seq_dir, heatmaps_dir, rows, stride=args.stride)
        print(f"  {seq_dir.name}: {n} usable triplets")

    df = pd.DataFrame(rows)
    index_path = output / "index.csv"
    df.to_csv(index_path, index=False)

    print(f"\nDone. {len(df)} total triplets written.")
    print(f"Index: {index_path}")
    print(f"Heatmaps: {heatmaps_dir}")

    print(f"\nNOTE: index.csv is NOT yet split into train/val.")
    print(f"When building the training script, split by the 'sequence' column")
    print(f"(e.g. group by sequence, hold out ~15% of sequences), same as the")
    print(f"YOLO pipeline -- never shuffle individual triplet rows across the split.")

if __name__ == "__main__":
    main()
