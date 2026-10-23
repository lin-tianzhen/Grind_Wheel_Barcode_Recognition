#!/usr/bin/env python3
"""Train the small U-Net on the one-shot augmented barcode dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from barcode_recog.config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH,
                                  NUM_WORKERS, POS_WEIGHT, SYNTH_DIR,
                                  TRAIN_HISTORY, WEIGHT_DECAY)
from barcode_recog.dataset import BarcodeSceneDataset
from barcode_recog.deep_model import BarcodeUNet, choose_device


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = union = 0.0
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(POS_WEIGHT, device=device)) + dice_loss(logits, y)
        loss_sum += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter += float((pred * y).sum())
        union += float(((pred + y) > 0).sum())
    return loss_sum / max(1, n), inter / max(1e-6, union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    device = choose_device()
    print(f"device = {device}")
    train_ds = BarcodeSceneDataset(SYNTH_DIR, "train", augment=True, size=args.size)
    val_ds = BarcodeSceneDataset(SYNTH_DIR, "val", augment=False, size=args.size)
    if args.limit:
        train_ds.items = train_ds.items[:args.limit]
        val_ds.items = val_ds.items[:max(4, args.limit // 4)]
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = BarcodeUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=args.lr * 0.05)
    pos_weight = torch.tensor(POS_WEIGHT, device=device)

    history = []
    best_iou = -1.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss = loss + dice_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += float(loss.detach()) * x.shape[0]
        train_loss = run_loss / len(train_ds)
        val_loss, val_iou = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_iou": val_iou,
                        "lr": sched.get_last_lr()[0]})
        mark = ""
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), MODEL_PATH)
            mark = " *saved"
        print(f"epoch {epoch:02d}/{args.epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  IoU {val_iou:.4f}{mark}")
    elapsed = time.time() - t_start
    TRAIN_HISTORY.write_text(json.dumps({
        "history": history, "best_val_iou": best_iou,
        "epochs": args.epochs, "params": n_params,
        "elapsed_sec": elapsed, "device": str(device),
    }, indent=2), encoding="utf-8")
    print(f"training finished in {elapsed/60:.1f} min; best val IoU={best_iou:.4f}")
    print(f"model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
