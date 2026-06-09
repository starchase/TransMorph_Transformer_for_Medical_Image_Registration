#!/usr/bin/env python3
"""Train TransMorph on OASIS with the repository's native PyTorch model API."""

import argparse
import copy
import csv
import glob
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import datasets, trans
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph


class LocalNCC(nn.Module):
    def __init__(self, window_size=9, eps=1e-5):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, target, prediction):
        win = self.window_size
        filt = target.new_ones((1, 1, win, win, win))
        padding = win // 2
        win_size = float(win ** 3)
        target_sum = F.conv3d(target, filt, padding=padding)
        pred_sum = F.conv3d(prediction, filt, padding=padding)
        target_sq_sum = F.conv3d(target * target, filt, padding=padding)
        pred_sq_sum = F.conv3d(prediction * prediction, filt, padding=padding)
        product_sum = F.conv3d(target * prediction, filt, padding=padding)
        target_mean = target_sum / win_size
        pred_mean = pred_sum / win_size
        cross = product_sum - pred_mean * target_sum - target_mean * pred_sum + target_mean * pred_mean * win_size
        target_var = target_sq_sum - 2 * target_mean * target_sum + target_mean.square() * win_size
        pred_var = pred_sq_sum - 2 * pred_mean * pred_sum + pred_mean.square() * win_size
        return -(cross.square() / (target_var * pred_var + self.eps)).mean()


def gradient_loss(flow):
    dx = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).square().mean()
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).square().mean()
    dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    return (dx + dy + dz) / 3.0


def dice_score(prediction, target, labels=range(1, 36)):
    scores = []
    for label in labels:
        pred_mask = prediction == label
        target_mask = target == label
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator > 0:
            scores.append((2.0 * (pred_mask & target_mask).sum().float() / denominator).item())
    return float(np.mean(scores)) if scores else 0.0


def create_model(config_name, image_size, device):
    config = copy.deepcopy(CONFIGS_TM[config_name])
    config.img_size = tuple(image_size)
    return TransMorph.TransMorph(config).to(device)


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return torch.cuda.amp.autocast()


def create_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, image_loss_fn, lambda_reg, device, scaler, amp_enabled):
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        source, target = (tensor.to(device, non_blocking=True).float() for tensor in batch[:2])
        for moving, fixed in ((source, target), (target, source)):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                warped, flow = model(torch.cat((moving, fixed), dim=1))
                image_loss = image_loss_fn(fixed, warped)
                regularization = gradient_loss(flow)
                loss = image_loss + lambda_reg * regularization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += [loss.item(), image_loss.item(), regularization.item()]
            steps += 1

    return tuple(totals / max(steps, 1))


@torch.no_grad()
def validate(model, label_transform, loader, image_loss_fn, lambda_reg, device):
    model.eval()
    losses = []
    dice_scores = []
    for source, target, source_seg, target_seg in tqdm(loader, desc="validate", leave=False):
        source = source.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        source_seg = source_seg.to(device, non_blocking=True).float()
        target_seg = target_seg.to(device, non_blocking=True).long()
        warped, flow = model(torch.cat((source, target), dim=1))
        losses.append((image_loss_fn(target, warped) + lambda_reg * gradient_loss(flow)).item())
        warped_seg = label_transform(source_seg, flow).round().long()
        dice_scores.append(dice_score(warped_seg, target_seg))
    return float(np.mean(losses)), float(np.mean(dice_scores))


def main():
    parser = argparse.ArgumentParser(description="Train native TransMorph on OASIS")
    parser.add_argument("--train-dir", default="/root/autodl-tmp/OASIS_L2R_2021_task03/All/")
    parser.add_argument("--val-dir", default="/root/autodl-tmp/OASIS_L2R_2021_task03/Test/")
    parser.add_argument("--output", default="/root/autodl-tmp/models/oasis_transmorph.pt")
    parser.add_argument("--transmorph-config", default="TransMorph-Large", choices=sorted(CONFIGS_TM))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.01)
    parser.add_argument("--loss", choices=("ncc", "mse"), default="ncc")
    parser.add_argument("--ncc-window", type=int, default=9)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_files = glob.glob(str(Path(args.train_dir) / "*.pkl"))
    val_files = glob.glob(str(Path(args.val_dir) / "*.pkl"))
    if not train_files or not val_files:
        parser.error("No OASIS .pkl files found in --train-dir or --val-dir.")

    composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.OASISBrainDataset(train_files, transforms=composed)
    val_set = datasets.OASISBrainInferDataset(val_files, transforms=composed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    image_size = tuple(train_set[0][0].shape[1:])
    model = create_model(args.transmorph_config, image_size, device)
    label_transform = TransMorph.SpatialTransformer(image_size, mode="nearest").to(device)
    image_loss_fn = LocalNCC(args.ncc_window).to(device) if args.loss == "ncc" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup, 1)),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = create_grad_scaler(amp_enabled)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"{output.stem}_log.csv"
    with log_path.open("w", newline="") as file:
        csv.writer(file).writerow(["epoch", "train_loss", "image_loss", "gradient_loss", "val_loss", "val_dice", "lr", "seconds"])

    best_dice = -1.0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, image_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, image_loss_fn, args.lambda_reg, device, scaler, amp_enabled
        )
        val_loss, val_dice = validate(model, label_transform, val_loader, image_loss_fn, args.lambda_reg, device)
        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_dice={val_dice:.6f} lr={lr:.2e} time={elapsed:.1f}s"
        )
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow([epoch + 1, train_loss, image_loss, reg_loss, val_loss, val_dice, lr, elapsed])

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output.parent / f"{output.stem}_best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output.parent / f"{output.stem}_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), output)
    print(f"Final model saved to {output}")


if __name__ == "__main__":
    main()
