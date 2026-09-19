
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracknet_model import TrackNet

INDEX_CSV    = r"C:\Users\asus\Downloads\NTI_Project\tracknet_dataset\index.csv"
OUTPUT_DIR   = r"C:\Users\asus\Downloads\NTI_Project\tracknet_output"

IMG_W, IMG_H = 1280, 720                                                     
BATCH_SIZE   = 2                                                       
EPOCHS       = 30
LR           = 1e-3
PATIENCE     = 10                                              
VAL_FRACTION = 0.15                                            
NUM_WORKERS  = 4                                       
SEED         = 42

class TrackNetDataset(Dataset):

    def __init__(self, rows, img_w=IMG_W, img_h=IMG_H):
        self.rows = rows
        self.img_w = img_w
        self.img_h = img_h

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        frames = []
        for key in ("frame_n2", "frame_n1", "frame_n"):
            img = cv2.imread(row[key])
            if img is None:
                raise FileNotFoundError(f"Cannot read frame: {row[key]}")
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        stacked = np.concatenate(frames, axis=2)             
        stacked = stacked.astype(np.float32) / 255.0
        stacked = stacked.transpose(2, 0, 1)                  

        heatmap = cv2.imread(row["heatmap"], cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise FileNotFoundError(f"Cannot read heatmap: {row['heatmap']}")
        heatmap = cv2.resize(heatmap, (self.img_w, self.img_h))
        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = heatmap[np.newaxis, ...]             

        orig_w = float(row["img_w"])
        orig_h = float(row["img_h"])
        gt_cx = float(row["cx"]) * (self.img_w / orig_w)
        gt_cy = float(row["cy"]) * (self.img_h / orig_h)

        return (
            torch.from_numpy(stacked),
            torch.from_numpy(heatmap),
            torch.tensor([gt_cx, gt_cy], dtype=torch.float32),
        )

def compute_metrics(logits, gt_coords, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu()                
    gt = gt_coords.cpu()          

    dists = []
    for i in range(probs.shape[0]):
        hm = probs[i, 0]          
        max_val = hm.max().item()

        if max_val < threshold:
            dists.append(float("inf"))
            continue

        peak_idx = hm.argmax().item()
        pred_y = peak_idx // hm.shape[1]
        pred_x = peak_idx % hm.shape[1]

        d = ((pred_x - gt[i, 0].item()) ** 2 + (pred_y - gt[i, 1].item()) ** 2) ** 0.5
        dists.append(d)

    return {"dists": dists, "n_samples": len(dists)}

def split_by_sequence(rows, val_fraction=VAL_FRACTION, seed=SEED):
    seqs = sorted(set(r["sequence"] for r in rows))
    random.seed(seed)
    random.shuffle(seqs)

    n_val = max(1, int(len(seqs) * val_fraction))
    val_seqs = set(seqs[:n_val])
    train_seqs = set(seqs[n_val:])

    train_rows = [r for r in rows if r["sequence"] in train_seqs]
    val_rows = [r for r in rows if r["sequence"] in val_seqs]

    return train_rows, val_rows, sorted(train_seqs), sorted(val_seqs)

def estimate_pos_weight(sample_rows, n_samples=100):
    rng = random.Random(SEED)
    subset = rng.sample(sample_rows, min(n_samples, len(sample_rows)))

    total_pos = 0
    total_neg = 0

    for row in subset:
        hm = cv2.imread(row["heatmap"], cv2.IMREAD_GRAYSCALE)
        if hm is None:
            continue
        hm = cv2.resize(hm, (IMG_W, IMG_H))
                                                                         
        pos = (hm > 25).sum()
        neg = hm.size - pos
        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        return 1.0                   

    ratio = total_neg / total_pos
                                         
    return min(ratio, 5000.0)

def save_sample_preds(model, val_dataset, device, output_dir, epoch, n=6):
    sample_dir = Path(output_dir) / "sample_preds"
    sample_dir.mkdir(exist_ok=True)

    model.eval()
    indices = list(range(min(n, len(val_dataset))))

    for i in indices:
        inp, gt_hm, gt_coords = val_dataset[i]
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = torch.sigmoid(model(inp.unsqueeze(0).to(device)))

        pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt_hm[0].numpy() * 255).astype(np.uint8)

        combined = np.hstack([gt_np, pred_np])

        cv2.putText(combined, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(combined, "Pred", (IMG_W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        cv2.imwrite(str(sample_dir / f"epoch{epoch:03d}_sample{i}.png"), combined)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(args.index_csv)
    rows = df.to_dict("records")
    print(f"Loaded {len(rows)} triplets from {args.index_csv}")

    train_rows, val_rows, train_seqs, val_seqs = split_by_sequence(rows)
    print(f"Train: {len(train_rows)} triplets from {len(train_seqs)} sequences")
    print(f"Val:   {len(val_rows)} triplets from {len(val_seqs)} sequences")
    print(f"Val sequences: {val_seqs}")

    train_ds = TrackNetDataset(train_rows)
    val_ds   = TrackNetDataset(val_rows)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = TrackNet(in_channels=9, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNet: {total_params:,} parameters")

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Successfully loaded checkpoint from epoch {ckpt.get('epoch')}. Resuming at epoch {start_epoch}!")

        best_pt_path = output_dir / "best.pt"
        if best_pt_path.exists():
            best_ckpt = torch.load(best_pt_path, map_location=device, weights_only=False)
            best_val_loss = best_ckpt.get("val_loss", float("inf"))
            print(f"Current all-time best val loss: {best_val_loss:.4f} (from epoch {best_ckpt.get('epoch')})")

    print("Estimating pos_weight from heatmap statistics...")
    pw = estimate_pos_weight(train_rows)
    print(f"pos_weight = {pw:.1f} (neg/pos pixel ratio)")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience,
    )

    scaler = torch.amp.GradScaler("cuda")

    log_path = output_dir / "training_log.csv"
    log_exists = log_path.exists() and args.resume is not None
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch", "train_loss", "val_loss", "val_mean_dist",
            "val_det_5", "val_det_10", "val_det_20", "lr", "time_min",
        ])

    patience_counter = 0
    start_time = time.time()

    print(f"\nStarting training: epochs {start_epoch} to {args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, resolution={IMG_W}x{IMG_H} (Gradient Clipping: 1.0)")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, (inputs, targets, gt_coords) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                                                                         
            loss = criterion(outputs.float(), targets.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg = train_loss_sum / train_batches
                print(f"  Epoch {epoch}/{args.epochs} - "
                      f"Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {avg:.4f}")

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_dists = []                                              
        total_val_samples = 0

        with torch.no_grad():
            for inputs, targets, gt_coords in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                              
                loss = criterion(outputs.float(), targets.float())

                val_loss_sum += loss.item()
                val_batches += 1

                metrics = compute_metrics(outputs, gt_coords)
                all_dists.extend(metrics["dists"])
                total_val_samples += metrics["n_samples"]

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        dists_arr = np.array(all_dists)
        finite = dists_arr[np.isfinite(dists_arr)]
        n_miss = int(np.isinf(dists_arr).sum())
        miss_rate = n_miss / max(len(dists_arr), 1) * 100

        if len(finite) > 0:
            avg_dist = float(finite.mean())
        else:
            avg_dist = float("inf")

        n_total = len(dists_arr)
        avg_det5  = float((dists_arr < 5).sum()  / max(n_total, 1) * 100)
        avg_det10 = float((dists_arr < 10).sum() / max(n_total, 1) * 100)
        avg_det20 = float((dists_arr < 20).sum() / max(n_total, 1) * 100)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = (time.time() - epoch_start) / 60

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dist: {avg_dist:.1f}px (miss {miss_rate:.1f}%) | "
              f"Det@10: {avg_det10:.1f}% | "
              f"Det@20: {avg_det20:.1f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}min")

        log_writer.writerow([
            epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
            f"{avg_dist:.2f}", f"{avg_det5:.1f}", f"{avg_det10:.1f}",
            f"{avg_det20:.1f}", f"{current_lr:.8f}", f"{epoch_time:.2f}",
        ])
        log_file.flush()

        scheduler.step(avg_val_loss)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_mean_dist": avg_dist,
            "val_det_10": avg_det10,
            "val_det_20": avg_det20,
            "img_w": IMG_W,
            "img_h": IMG_H,
        }

        torch.save(checkpoint, str(output_dir / "last.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint, str(output_dir / "best.pt"))
            print(f"  ** New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        if epoch % 5 == 0 or epoch == 1:
            save_sample_preds(model, val_ds, device, output_dir, epoch)

    total_time = (time.time() - start_time) / 60
    log_file.close()

    print("\n" + "=" * 80)
    print(f"Training complete in {total_time:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir / 'best.pt'}, {output_dir / 'last.pt'}")
    print(f"Training log: {log_path}")
    print(f"Sample predictions: {output_dir / 'sample_preds'}")

    print("\n── Final validation with best model ──")
    best_ckpt = torch.load(str(output_dir / "best.pt"), map_location=device,
                           weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    final_metrics = {"mean_dist": [], "det_5": [], "det_10": [], "det_20": []}
    with torch.no_grad():
        for inputs, targets, gt_coords in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
            m = compute_metrics(outputs, gt_coords)
            for k, v in m.items():
                final_metrics[k].append(v)

    print(f"Mean pixel distance: {np.mean(final_metrics['mean_dist']):.1f}px")
    print(f"Detection rate @5px:  {np.mean(final_metrics['det_5']):.1f}%")
    print(f"Detection rate @10px: {np.mean(final_metrics['det_10']):.1f}%")
    print(f"Detection rate @20px: {np.mean(final_metrics['det_20']):.1f}%")
    print(f"(Best model from epoch {best_ckpt['epoch']})")

def main():
    parser = argparse.ArgumentParser(description="Train TrackNet for ball tracking")
    parser.add_argument("--index-csv", default=INDEX_CSV,
                        help="Path to index.csv from build_tracknet_dataset.py")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    train(args)

if __name__ == "__main__":
    main()
