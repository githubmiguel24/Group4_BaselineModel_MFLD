#!/usr/bin/env python3
import argparse, os, time
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
torch.backends.cudnn.benchmark = True

import config
from dataset import build_dataloaders
from loss import MultiTaskLoss
from metrics import KeypointEvaluator
from model import MFLDNet
from utils import save_checkpoint, load_checkpoint, set_seed, count_parameters, model_size_mb


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    total_loss = coord_loss = hmap_loss = 0.0
    n_batches = len(loader)
    pbar = tqdm(loader, desc=f"  Train E{epoch:03d}", leave=False)
    for batch in pbar:
        images, gt_hm, gt_coords, vis, paths, bbox = batch   # bbox ignored in training
        images = images.to(device, non_blocking=True)
        gt_hm   = gt_hm.to(device, non_blocking=True)
        gt_coords = gt_coords.to(device, non_blocking=True)
        vis      = vis.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            pred_hm, pred_co = model(images)
            loss, c_loss, h_loss = criterion(pred_hm, pred_co, gt_hm, gt_coords, vis)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        coord_loss += c_loss.item()
        hmap_loss  += h_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", coord=f"{c_loss.item():.4f}", hmap=f"{h_loss.item():.4f}")
    n = max(n_batches, 1)
    return {"total": total_loss/n, "coord": coord_loss/n, "hmap": hmap_loss/n}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = coord_loss = hmap_loss = 0.0
    evaluator = KeypointEvaluator()
    n_batches = len(loader)
    pbar = tqdm(loader, desc=f"  Valid E{epoch:03d}", leave=False)
    for batch in pbar:
        images, gt_hm, gt_coords, vis, paths, bbox = batch   # bbox: list of tuples or None
        images = images.to(device, non_blocking=True)
        gt_hm   = gt_hm.to(device, non_blocking=True)
        gt_coords = gt_coords.to(device, non_blocking=True)
        vis      = vis.to(device, non_blocking=True)

        with autocast():
            pred_hm, pred_co = model(images)
            loss, c_loss, h_loss = criterion(pred_hm, pred_co, gt_hm, gt_coords, vis)
        total_loss += loss.item()
        coord_loss += c_loss.item()
        hmap_loss  += h_loss.item()

        # Pass bbox to evaluator
        evaluator.update(pred_hm, pred_co, gt_hm, gt_coords, vis, bbox=bbox)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n = max(n_batches, 1)
    metrics = evaluator.compute()
    return {"total": total_loss/n, "coord": coord_loss/n, "hmap": hmap_loss/n, **metrics}


def train_model(
    image_dir=config.IMAGE_DIR, annot_file=config.ANNOT_FILE,
    ckpt_dir=config.CHECKPOINT_DIR, log_dir=config.LOG_DIR,
    epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
    lr=config.LEARNING_RATE, resume=None, num_workers=4, device_str="auto"
):
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str=="auto" else torch.device(device_str)
    print(f"\n{'='*60}\n  Model  : MFLD-net\n  Device : {device}")

    train_loader, val_loader, _ = build_dataloaders(image_dir, annot_file, batch_size, num_workers)
    model = MFLDNet().to(device)
    print(f"  Params : {count_parameters(model)/1e6:.2f} M   Size: {model_size_mb(model):.1f} MB")

    criterion = MultiTaskLoss(alpha=config.MULTITASK_ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                           eps=config.ADAM_EPS, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=config.LR_DECAY_STEP, gamma=config.LR_DECAY_GAMMA)
    scaler = GradScaler()

    start_epoch, best_loss = 0, float("inf")
    if resume and os.path.isfile(resume):
        start_epoch, best_loss = load_checkpoint(resume, model, optimizer)
        print(f"  Resumed from {resume} (epoch {start_epoch})")

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "mfld_net"))
    best_val_metrics = {}
    history = {"train": [], "val": []}

    for epoch in range(start_epoch+1, epochs+1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)
        if True:
            val_metrics = validate(model, val_loader, criterion, device, epoch)
        else:
            val_metrics = {"total": float("nan")}
        scheduler.step()
        elapsed = time.time()-t0
        print(f"  E{epoch:03d}/{epochs}  train[total={train_metrics['total']:.4f} coord={train_metrics['coord']:.4f} hmap={train_metrics['hmap']:.4f}]  "
              f"val[total={val_metrics['total']:.4f} AP={val_metrics.get('AP', float('nan')):.3f}]  {elapsed:.1f}s")

        for split, met in [("train", train_metrics), ("val", val_metrics)]:
            for k,v in met.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{split}/{k}", v, epoch)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if epoch % 5 == 0 or epoch == epochs:
            is_best = val_metrics["total"] < best_loss
            if is_best:
                best_loss = val_metrics["total"]
                best_val_metrics = val_metrics.copy()
        else:
            is_best = False

        ckpt_path = os.path.join(ckpt_dir, f"mfld_net_epoch_{epoch:03d}.pth")
        save_checkpoint({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
            "val_metrics": val_metrics,
        }, path=ckpt_path, is_best=is_best)

    writer.close()
    print("\n  ── Best validation results for [MFLD-net] ──")
    KeypointEvaluator.print_results(best_val_metrics)
    return {"best_loss": best_loss, "best_metrics": best_val_metrics, "history": history}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=config.IMAGE_DIR)
    parser.add_argument("--annot_file", default=config.ANNOT_FILE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(
        image_dir=args.image_dir, annot_file=args.annot_file,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, resume=args.resume,
        num_workers=args.num_workers, device_str=args.device,
    )