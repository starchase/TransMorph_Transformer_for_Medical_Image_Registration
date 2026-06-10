#!/usr/bin/env python3
"""Train TransMorph for multimodal CT-to-MR registration."""

import argparse
import copy
import csv
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph


def is_nifti(path):
    return path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))


def list_nifti(directory):
    directory = Path(directory) if directory else None
    return sorted(path for path in directory.iterdir() if is_nifti(path)) if directory and directory.exists() else []


def case_key(path):
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    if name.endswith("_0000") or name.endswith("_0001"):
        name = name[:-5]
    return name


def pair_files(source_dir, target_dir):
    source = {case_key(path): path for path in list_nifti(source_dir)}
    target = {case_key(path): path for path in list_nifti(target_dir)}
    return [(source[key], target[key]) for key in sorted(source.keys() & target.keys())]


def load_volume(path, label=False):
    array = nib.load(str(path)).get_fdata()
    if label:
        array = np.rint(array).astype(np.int16)
    else:
        array = np.nan_to_num(array.astype(np.float32))
        low, high = float(array.min()), float(array.max())
        if high > low:
            array = (array - low) / (high - low)
    return torch.from_numpy(array).unsqueeze(0)


class MultimodalTrainDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, paired_ct_dir=None, paired_mr_dir=None, samples_per_epoch=100):
        self.ct_files = list_nifti(ct_dir)
        self.mr_files = list_nifti(mr_dir)
        self.fixed_pairs = pair_files(paired_ct_dir, paired_mr_dir)
        self.samples_per_epoch = samples_per_epoch
        if not self.ct_files or not self.mr_files:
            raise ValueError("Training CT and MR directories must contain NIfTI files.")

    def __len__(self):
        return self.samples_per_epoch + len(self.fixed_pairs)

    def __getitem__(self, index):
        if index < self.samples_per_epoch:
            ct_path, mr_path = random.choice(self.ct_files), random.choice(self.mr_files)
        else:
            ct_path, mr_path = self.fixed_pairs[index - self.samples_per_epoch]
        return {"source": load_volume(ct_path), "target": load_volume(mr_path)}


class MultimodalValidationDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, ct_label_dir=None, mr_label_dir=None):
        self.pairs = pair_files(ct_dir, mr_dir)
        self.ct_labels = {case_key(path): path for path in list_nifti(ct_label_dir)}
        self.mr_labels = {case_key(path): path for path in list_nifti(mr_label_dir)}
        if not self.pairs:
            raise ValueError("Validation CT and MR directories do not contain matching cases.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ct_path, mr_path = self.pairs[index]
        key = case_key(ct_path)
        sample = {"source": load_volume(ct_path), "target": load_volume(mr_path)}
        if key in self.ct_labels and key in self.mr_labels:
            sample["source_label"] = load_volume(self.ct_labels[key], label=True)
            sample["target_label"] = load_volume(self.mr_labels[key], label=True)
        return sample


class LocalNCC(nn.Module):
    def __init__(self, window_size=9, eps=1e-5):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, target, prediction, mask=None):
        if mask is not None:
            target, prediction = target * mask, prediction * mask
        win = self.window_size
        filt = target.new_ones((1, 1, win, win, win))
        padding = win // 2
        count = float(win ** 3)
        target_sum = F.conv3d(target, filt, padding=padding)
        pred_sum = F.conv3d(prediction, filt, padding=padding)
        target_sq_sum = F.conv3d(target.square(), filt, padding=padding)
        pred_sq_sum = F.conv3d(prediction.square(), filt, padding=padding)
        product_sum = F.conv3d(target * prediction, filt, padding=padding)
        target_mean, pred_mean = target_sum / count, pred_sum / count
        cross = product_sum - pred_mean * target_sum - target_mean * pred_sum + target_mean * pred_mean * count
        target_var = target_sq_sum - 2 * target_mean * target_sum + target_mean.square() * count
        pred_var = pred_sq_sum - 2 * pred_mean * pred_sum + pred_mean.square() * count
        return -(cross.square() / (target_var * pred_var + self.eps)).mean()


class MINDLoss(nn.Module):
    def __init__(self, radius=2, dilation=2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation

    def descriptor(self, image):
        shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        padded = F.pad(image, (self.dilation,) * 6, mode="replicate")
        descriptors = []
        d, h, w = image.shape[2:]
        for dz, dy, dx in shifts:
            z = self.dilation + dz * self.dilation
            y = self.dilation + dy * self.dilation
            x = self.dilation + dx * self.dilation
            shifted = padded[:, :, z:z + d, y:y + h, x:x + w]
            descriptors.append(F.avg_pool3d((image - shifted).square(), 2 * self.radius + 1, stride=1, padding=self.radius))
        descriptor = torch.cat(descriptors, dim=1)
        variance = descriptor.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.exp(-descriptor / variance)

    def forward(self, target, prediction, mask=None):
        difference = (self.descriptor(target) - self.descriptor(prediction)).square()
        if mask is not None:
            difference = difference * mask
        return difference.mean()


class ImageLoss(nn.Module):
    def __init__(self, name, ncc_window=9, mind_radius=2, mind_dilation=2):
        super().__init__()
        self.name = name
        self.ncc = LocalNCC(ncc_window)
        self.mind = MINDLoss(mind_radius, mind_dilation)

    def forward(self, target, prediction, mask=None):
        if self.name == "ncc":
            return self.ncc(target, prediction, mask)
        if self.name == "mind":
            return self.mind(target, prediction, mask)
        difference = (target - prediction).square()
        if mask is not None:
            return (difference * mask).sum() / mask.sum().clamp_min(1.0)
        return difference.mean()


def gradient_loss(flow):
    values = [
        (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).square().mean(),
        (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).square().mean(),
        (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean(),
    ]
    return sum(values) / 3.0


def bending_loss(flow):
    values = [
        (flow[:, :, 2:, :, :] - 2 * flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]).square().mean(),
        (flow[:, :, :, 2:, :] - 2 * flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]).square().mean(),
        (flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]).square().mean(),
    ]
    return sum(values) / 3.0


def foreground_mask(image):
    threshold = image.amin(dim=(2, 3, 4), keepdim=True) + 1e-3
    return (image > threshold).float()


def dice_score(prediction, target):
    labels = torch.unique(torch.cat((prediction.flatten(), target.flatten())))
    scores = []
    for label in labels:
        if label.item() == 0:
            continue
        pred_mask, target_mask = prediction == label, target == label
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator > 0:
            scores.append((2.0 * (pred_mask & target_mask).sum().float() / denominator).item())
    return float(np.mean(scores)) if scores else float("nan")


def create_model(config_name, image_size, integration_steps, device):
    config = copy.deepcopy(CONFIGS_TM[config_name])
    config.img_size = tuple(image_size)
    config.integration_steps = integration_steps
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


def train_epoch(model, loader, optimizer, image_loss_fn, lambda_reg, bend_weight, mask_mode, device, scaler, amp_enabled):
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        source = batch["source"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        mask = foreground_mask(target) if mask_mode == "auto" else None
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            warped, flow = model(torch.cat((source, target), dim=1))
            image_loss = image_loss_fn(target, warped, mask)
            smoothness = gradient_loss(flow)
            bending = bending_loss(flow) if bend_weight > 0 else flow.new_tensor(0.0)
            loss = image_loss + lambda_reg * smoothness + bend_weight * bending
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals += [loss.item(), image_loss.item(), smoothness.item(), bending.item()]
        steps += 1
    return tuple(totals / max(steps, 1))


@torch.no_grad()
def validate(model, label_transform, loader, image_loss_fn, lambda_reg, bend_weight, mask_mode, device):
    model.eval()
    losses, dice_scores = [], []
    for batch in tqdm(loader, desc="validate", leave=False):
        source = batch["source"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        warped, flow = model(torch.cat((source, target), dim=1))
        mask = foreground_mask(target) if mask_mode == "auto" else None
        loss = image_loss_fn(target, warped, mask) + lambda_reg * gradient_loss(flow)
        if bend_weight > 0:
            loss = loss + bend_weight * bending_loss(flow)
        losses.append(loss.item())
        if "source_label" in batch:
            source_label = batch["source_label"].to(device).float()
            target_label = batch["target_label"].to(device).long()
            warped_label = label_transform(source_label, flow).round().long()
            dice_scores.append(dice_score(warped_label, target_label))
    return float(np.mean(losses)), float(np.nanmean(dice_scores)) if dice_scores else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Train native TransMorph for multimodal CT-to-MR registration")
    parser.add_argument("--ct-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/train/images/ct")
    parser.add_argument("--mr-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/train/images/mr")
    parser.add_argument("--paired-ct-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images/ct")
    parser.add_argument("--paired-mr-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images/mr")
    parser.add_argument("--ct-val-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/images/ct")
    parser.add_argument("--mr-val-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/images/mr")
    parser.add_argument("--ct-val-label-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/labels/ct")
    parser.add_argument("--mr-val-label-dir", default="/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/labels/mr")
    parser.add_argument("--output", default="/root/autodl-tmp/models/multimodal_transmorph.pt")
    parser.add_argument("--transmorph-config", default="TransMorph", choices=sorted(CONFIGS_TM))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.01)
    parser.add_argument("--bend-weight", type=float, default=0.0)
    parser.add_argument("--image-loss", choices=("ncc", "mse", "mind"), default="mind")
    parser.add_argument("--ncc-window", type=int, default=9)
    parser.add_argument("--mind-radius", type=int, default=2)
    parser.add_argument("--mind-dilation", type=int, default=2)
    parser.add_argument("--loss-mask-mode", choices=("auto", "none"), default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--integration-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    if args.integration_steps < 0:
        parser.error("--integration-steps must be non-negative.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_set = MultimodalTrainDataset(
        args.ct_dir, args.mr_dir, args.paired_ct_dir, args.paired_mr_dir, args.samples_per_epoch
    )
    val_set = MultimodalValidationDataset(
        args.ct_val_dir, args.mr_val_dir, args.ct_val_label_dir, args.mr_val_label_dir
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    image_size = tuple(train_set[0]["source"].shape[1:])
    model = create_model(args.transmorph_config, image_size, args.integration_steps, device)
    label_transform = TransMorph.SpatialTransformer(image_size, mode="nearest").to(device)
    image_loss_fn = ImageLoss(args.image_loss, args.ncc_window, args.mind_radius, args.mind_dilation).to(device)
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
        csv.writer(file).writerow(["epoch", "train_loss", "image_loss", "gradient_loss", "bending_loss", "val_loss", "val_dice", "lr", "seconds"])

    best_loss = float("inf")
    for epoch in range(args.epochs):
        start = time.time()
        train_values = train_epoch(
            model, train_loader, optimizer, image_loss_fn, args.lambda_reg, args.bend_weight,
            args.loss_mask_mode, device, scaler, amp_enabled
        )
        val_loss, val_dice = validate(
            model, label_transform, val_loader, image_loss_fn, args.lambda_reg, args.bend_weight, args.loss_mask_mode, device
        )
        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} loss={train_values[0]:.6f} "
            f"val_loss={val_loss:.6f} val_dice={val_dice:.6f} lr={lr:.2e} time={elapsed:.1f}s"
        )
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow([epoch + 1, *train_values, val_loss, val_dice, lr, elapsed])
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output.parent / f"{output.stem}_best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output.parent / f"{output.stem}_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), output)
    print(f"Final model saved to {output}")


if __name__ == "__main__":
    main()
