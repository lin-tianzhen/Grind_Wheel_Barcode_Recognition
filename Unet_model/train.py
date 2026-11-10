"""训练与验证逻辑."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import MODEL_DIR, REAL_DIR, SYNTH_DIR
from .dataset import BarcodeDataset
from .losses import CombinedLoss
from .model import UNet


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    iou_sum = 0.0
    n_pos = 0
    fp = 0
    n_neg = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * y).sum(dim=(1, 2, 3))
        union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
        has = y.sum(dim=(1, 2, 3)) > 0
        iou = inter / union.clamp(min=1)
        if has.any():
            iou_sum += iou[has].sum().item()
            n_pos += int(has.sum().item())
        if (~has).any():
            neg_pred = preds[~has].sum(dim=(1, 2, 3)) > 0
            fp += int(neg_pred.sum().item())
            n_neg += int((~has).sum().item())
        n += x.size(0)
    mean_iou = iou_sum / max(1, n_pos)
    fp_rate = (fp / n_neg) if n_neg else 0.0
    return total_loss / max(1, n), mean_iou, fp_rate


def train(args):
    device = get_device()
    print(f"设备: {device}")

    if (REAL_DIR / "train_manifest.json").exists():
        data_root = REAL_DIR
    else:
        data_root = SYNTH_DIR
    print(f"数据集目录: {data_root}")

    train_ds = BarcodeDataset(data_root, "train", size=args.size, augment=True)
    val_ds = BarcodeDataset(data_root, "val", size=args.size, augment=False,balance=False)
    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

    cover = [float((m > 127).mean()) for _, m in train_ds._cache] if train_ds._cache else []
    if cover:
        print(f"训练 mask 覆盖率: 中位数={np.median(cover):.3f} "
              f"范围=[{min(cover):.3f}, {max(cover):.3f}]")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=pin_mem)

    model = UNet(in_ch=1, base_ch=args.base_ch, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net 参数量: {n_params / 1e6:.2f}M")

    criterion = CombinedLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    warmup = max(8, args.epochs // 15)

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_iou = 0.0
    patience_counter = 0
    patience = args.patience
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    model_path = MODEL_DIR / "unet_barcode.pt"

    print(f"\n开始训练: {args.epochs} epochs, batch={args.batch}, "
          f"lr={args.lr}, size={args.size}, dropout={args.dropout}")
    print(f"学习率: 线性 warmup {warmup} 轮 + 余弦退火; 早停 patience={patience}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, fp_rate = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        marker = ""
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_loss": val_loss,
                "args": vars(args),
                "train_size": args.size,
            }, str(model_path))
            marker = " *"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_iou={val_iou:.4f} fp_rate={fp_rate:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
              f"best_iou={best_iou:.4f} | "
              f"{elapsed:.1f}s{marker}")

        if patience_counter >= patience:
            print(f"\n早停触发: val_iou 连续 {patience} 轮未提升")
            break

    hist_path = MODEL_DIR / "train_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"训练完成! 最佳 val IoU = {best_iou:.4f}")
    print(f"模型已保存: {model_path}")
    print(f"训练历史: {hist_path}")