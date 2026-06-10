#!/usr/bin/env python3
"""Train TransMorph on OASIS with the repository's native PyTorch model API."""

import argparse
import copy
import csv
import datetime
import glob
import os
import time
from contextlib import nullcontext
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.spatial import cKDTree
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
    def __init__(self, window_size=9, eps=1e-3):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, target, prediction):
        target = target.float()
        prediction = prediction.float()
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
        ncc = cross.square() / (target_var * pred_var + self.eps)
        return -ncc.mean()


def gradient_loss(flow):
    flow = flow.float()
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


def compute_hd95(ground_truth, prediction):
    if not ground_truth.any() or not prediction.any():
        return np.nan
    pred_border = prediction ^ scipy.ndimage.binary_erosion(prediction)
    gt_border = ground_truth ^ scipy.ndimage.binary_erosion(ground_truth)
    pred_points = np.argwhere(pred_border)
    gt_points = np.argwhere(gt_border)
    if not len(pred_points) or not len(gt_points):
        return np.nan
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    return max(
        np.percentile(gt_tree.query(pred_points)[0], 95),
        np.percentile(pred_tree.query(gt_points)[0], 95),
    )


def jacobian_determinant(flow):
    # flow: (C, H, W, D) where C==ndim (3)
    # convert to (H, W, D, C)
    displacement = np.moveaxis(flow, 0, -1)
    # create identity grid and add to displacement to get mapping coordinates
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in displacement.shape[:-1]], indexing="ij"), axis=-1)
    field = displacement + grid

    # For each vector component, compute partial derivatives wrt spatial axes
    # grads_c is list of 3 arrays (d(component)/dx, d(component)/dy, d(component)/dz)
    jac_components = []
    for c in range(field.shape[-1]):
        grads = np.gradient(field[..., c])  # returns tuple (d/dx, d/dy, d/dz)
        # stack to (H, W, D, 3) where last dim are derivatives for this component
        jac_components.append(np.stack(grads, axis=-1))

    # Stack components to shape (H, W, D, 3, 3) where [i,j] = d(component_i)/d(coord_j)
    jac = np.stack(jac_components, axis=-2)

    # Determinant over last two axes -> (H, W, D)
    return np.linalg.det(jac)


@torch.no_grad()
def save_qualitative_results(model, label_transform, dataset, output_dir, epoch, device):
    source, target, source_seg, target_seg = dataset[min(9, len(dataset) - 1)]
    source = source.unsqueeze(0).to(device).float()
    target = target.unsqueeze(0).to(device).float()
    source_seg = source_seg.unsqueeze(0).to(device).float()
    target_seg = target_seg.unsqueeze(0).to(device).long()
    warped, flow = model(torch.cat((source, target), dim=1))
    warped_seg = label_transform(source_seg, flow).round().long()
    z_index = source.shape[-1] // 2
    tensors = (source, target, warped, source_seg, target_seg, warped_seg)
    titles = ("Source", "Target", "Warped Source", "Source Label", "Target Label", "Warped Label")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, tensor, title in zip(axes.flat, tensors, titles):
        image = np.rot90(tensor[0, 0, :, :, z_index].detach().cpu().numpy(), -1)
        axis.imshow(image, cmap="gray" if "Label" not in title else "tab20")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Epoch {epoch} - Sample default")
    fig.tight_layout()
    fig.savefig(output_dir / f"vis_epoch_{epoch:04d}_default.png")
    plt.close(fig)


def create_model(config_name, image_size, integration_steps, device):
    config = copy.deepcopy(CONFIGS_TM[config_name])
    config.img_size = tuple(image_size)
    config.integration_steps = integration_steps
    return TransMorph.TransMorph(config).to(device)


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        supports_bfloat16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
        return torch.autocast(device_type=device.type, dtype=dtype)
    return torch.cuda.amp.autocast()


def create_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, image_loss_fn, lambda_reg, bidirectional, accumulation_steps, device, scaler, amp_enabled):
    model.train()
    totals = np.zeros(5, dtype=np.float64)
    steps = 0
    pending_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="train", leave=False):
        source, target = (tensor.to(device, non_blocking=True).float() for tensor in batch[:2])
        pairs = ((source, target), (target, source)) if bidirectional else ((source, target),)
        for moving, fixed in pairs:
            with autocast_context(device, amp_enabled):
                warped, flow = model(torch.cat((moving, fixed), dim=1))
            image_loss = image_loss_fn(fixed, warped)
            regularization = gradient_loss(flow)
            loss = image_loss + lambda_reg * regularization
            scaler.scale(loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
            totals += [
                loss.item(), image_loss.item(), regularization.item(),
                flow.detach().float().abs().mean().item(), flow.detach().float().abs().max().item(),
            ]
            steps += 1

    if pending_steps:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    return tuple(totals / max(steps, 1))


@torch.no_grad()
def validate(model, label_transform, loader, image_loss_fn, lambda_reg, device, compute_extra=False):
    model.eval()
    losses = []
    dice_scores = []
    initial_dice_scores = []
    flow_means = []
    flow_maxes = []
    hd95_scores = []
    jac_scores = []
    magnitude_scores = []
    for source, target, source_seg, target_seg in tqdm(loader, desc="validate", leave=False):
        source = source.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        source_seg = source_seg.to(device, non_blocking=True).float()
        target_seg = target_seg.to(device, non_blocking=True).long()
        warped, flow = model(torch.cat((source, target), dim=1))
        image_loss = image_loss_fn(target, warped)
        losses.append((image_loss + lambda_reg * gradient_loss(flow)).item())
        warped_seg = label_transform(source_seg, flow).round().long()
        dice_scores.append(dice_score(warped_seg, target_seg))
        initial_dice_scores.append(dice_score(source_seg.long(), target_seg))
        flow_means.append(flow.float().abs().mean().item())
        flow_maxes.append(flow.float().abs().max().item())
        if compute_extra:
            magnitude_scores.append(torch.linalg.vector_norm(flow.float(), dim=1).mean().item())
            flow_np = flow[0].float().cpu().numpy()
            target_np = target[0, 0].cpu().numpy()
            jac_det = jacobian_determinant(flow_np)
            jac_scores.append(float(((jac_det <= 0) & (target_np > 0.01)).sum() / max((target_np > 0.01).sum(), 1)))
            warped_np = warped_seg[0, 0].cpu().numpy()
            target_seg_np = target_seg[0, 0].cpu().numpy()
            case_hd95 = [
                compute_hd95(target_seg_np == label, warped_np == label)
                for label in range(1, 36)
                if np.any(target_seg_np == label) and np.any(warped_np == label)
            ]
            hd95_scores.append(float(np.nanmean(case_hd95)) if case_hd95 else np.nan)
    return (
        float(np.mean(losses)),
        float(np.mean(dice_scores)),
        float(np.mean(initial_dice_scores)),
        float(np.mean(flow_means)),
        float(np.max(flow_maxes)),
        float(np.nanmean(hd95_scores)) if compute_extra else np.nan,
        float(np.mean(jac_scores)) if compute_extra else np.nan,
        float(np.mean(magnitude_scores)) if compute_extra else np.nan,
    )


def main():
    parser = argparse.ArgumentParser(description="Train native TransMorph on OASIS")
    parser.add_argument("--train-dir", default="/root/autodl-tmp/OASIS_L2R_2021_task03/All/")
    parser.add_argument("--val-dir", default="/root/autodl-tmp/OASIS_L2R_2021_task03/Test/")
    parser.add_argument("--output", default="/root/autodl-tmp/models/oasis_transmorph.pt")
    parser.add_argument("--transmorph-config", default="TransMorph", choices=sorted(CONFIGS_TM))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=1.2)
    parser.add_argument("--loss", choices=("ncc", "mse"), default="ncc")
    parser.add_argument("--ncc-window", type=int, default=9)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--vis-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--warm-start", type=int, default=10)
    parser.add_argument("--integration-steps", type=int, default=5)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    if args.integration_steps < 0:
        parser.error("--integration-steps must be non-negative.")
    if args.accumulation_steps < 1:
        parser.error("--accumulation-steps must be positive.")

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
    model = create_model(args.transmorph_config, image_size, args.integration_steps, device)
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
    input_output_path = Path(args.output)
    timestamp = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y%m%d_%H%M%S")
    run_dir = input_output_path.parent / f"{input_output_path.stem}_{timestamp}"
    output = run_dir / f"{input_output_path.stem}{input_output_path.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for this run: {output.parent}")

    log_path = output.parent / "train_log.csv"
    config_path = output.parent / "config.txt"
    total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with config_path.open("w") as file:
        file.write("Training Configuration:\n")
        file.write(f"Device: {device}\n")
        file.write(f"Model Architecture: {args.transmorph_config}\n")
        file.write(f"Total Parameters: {total_params:,}\n")
        file.write(f"Epochs: {args.epochs}\n")
        file.write(f"Batch Size: {args.batch_size}\n")
        file.write(f"Lambda: {args.lambda_reg}\n")
        file.write(f"Loss: {args.loss}\n")
        file.write("Use Mask: False\n")
        file.write(f"Integration Steps: {args.integration_steps}\n")
        file.write(f"Bidirectional Training: {args.bidirectional}\n")
        file.write(f"Accumulation Steps: {args.accumulation_steps}\n")
        file.write(f"LR: {args.lr}\n")
        file.write(f"Arguments: {vars(args)}\n")
    with log_path.open("w", newline="") as file:
        csv.writer(file).writerow([
            "epoch", "train_loss", "train_img_loss", "train_grad_loss",
            "train_feature_edge_loss", "val_dsc", "val_hd95", "val_jac", "val_mag",
        ])

    best_dice = -1.0
    initial_values = validate(
        model, label_transform, val_loader, image_loss_fn, args.lambda_reg, device
    )
    initial_val_loss, initial_val_dice, initial_dice, initial_flow_mean, initial_flow_max = initial_values[:5]
    print(
        f"Initial baseline val_loss={initial_val_loss:.6f} val_dice={initial_val_dice:.6f} "
        f"gain={initial_val_dice - initial_dice:+.6f} "
        f"flow={initial_flow_mean:.4f}/{initial_flow_max:.4f}"
    )

    loss_history = []
    val_dsc_history = []
    epoch_times = []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, image_loss, reg_loss, train_flow_mean, train_flow_max = train_epoch(
            model, train_loader, optimizer, image_loss_fn, args.lambda_reg, args.bidirectional,
            args.accumulation_steps, device, scaler, amp_enabled
        )
        val_values = validate(
            model, label_transform, val_loader, image_loss_fn, args.lambda_reg, device
        )
        val_loss, val_dice, initial_dice, val_flow_mean, val_flow_max = val_values[:5]
        dice_gain = val_dice - initial_dice
        is_new_best = val_dice > best_dice
        epoch_num = epoch + 1
        compute_extra = epoch_num in (1, 3, 5, 7) or epoch_num % 10 == 0 or (is_new_best and val_dice > 0.77)
        if compute_extra:
            extra_values = validate(
                model, label_transform, val_loader, image_loss_fn, args.lambda_reg, device, compute_extra=True
            )
            val_hd95, val_jac, val_mag = extra_values[5:]
        else:
            val_hd95, val_jac, val_mag = np.nan, np.nan, np.nan
        scheduler.step()
        elapsed = time.time() - start
        epoch_times.append(elapsed)
        loss_history.append(train_loss)
        val_dsc_history.append(val_dice)
        lr = optimizer.param_groups[0]["lr"]
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        metrics = (
            f", HD95: {val_hd95:.2f}, Jac: {val_jac:.4f}, Mag: {val_mag:.4f}"
            if compute_extra else ""
        )
        print(
            f"Epoch {epoch_num}/{args.epochs}, Loss: {train_loss:.6f}, Img: {image_loss:.6f}, "
            f"Grad: {reg_loss:.6f}, FeatureEdge: 0.000000, Val DSC: {val_dice:.6f}{metrics}, "
            f"LR: {lr:.6f}, Time: {elapsed:.2f}s, Peak: {peak_memory:.2f}MB, "
            f"Gain: {dice_gain:+.6f}, Flow: {val_flow_mean:.4f}/{val_flow_max:.4f}"
        )
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow([
                epoch_num, f"{train_loss:.6f}", f"{image_loss:.6f}", f"{reg_loss:.6f}", "0.000000",
                f"{val_dice:.6f}", f"{val_hd95:.2f}" if compute_extra else "",
                f"{val_jac:.6f}" if compute_extra else "", f"{val_mag:.6f}" if compute_extra else "",
            ])

        if compute_extra and args.vis_every != 0:
            save_qualitative_results(model, label_transform, val_set, output.parent, epoch_num, device)

        if is_new_best:
            best_dice = val_dice
            torch.save(model.state_dict(), output.parent / f"{output.stem}_best.pt")
            print(f"Saved new best model with DSC: {best_dice:.6f} (HD95: {val_hd95:.2f}, Jac: {val_jac:.4f})")
        if args.save_every > 0 and epoch_num % args.save_every == 0:
            checkpoint = output.parent / f"{output.stem}_epoch{epoch_num}.pt"
            torch.save(model.state_dict(), checkpoint)
            print(f"Checkpoint saved to {checkpoint}")

        if epoch_num % 10 == 0:
            fig, axis_loss = plt.subplots(figsize=(10, 6))
            axis_loss.plot(range(1, epoch_num + 1), loss_history, color="tab:red", label="Train Loss")
            axis_loss.set_xlabel("Epoch")
            axis_loss.set_ylabel("Train Loss", color="tab:red")
            axis_dice = axis_loss.twinx()
            axis_dice.plot(range(1, epoch_num + 1), val_dsc_history, color="tab:blue", label="Val DSC")
            axis_dice.set_ylabel("Val DSC", color="tab:blue")
            fig.tight_layout()
            fig.savefig(output.parent / f"learning_curves_epoch{epoch_num}.png", dpi=150)
            plt.close(fig)

        if len(loss_history) >= args.warm_start + args.patience + 1:
            recent = loss_history[-args.patience:]
            best_past = min(loss_history[:-args.patience])
            if all(max(best_past - loss, 0) < args.threshold for loss in recent):
                print(f"Early stopping at epoch {epoch_num}")
                break

    torch.save(model.state_dict(), output)
    print(f"Final model saved to {output}")
    with config_path.open("a") as file:
        file.write(f"Average Epoch Time: {np.mean(epoch_times):.2f} s\n")
        file.write(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0:.2f} MB\n")


if __name__ == "__main__":
    main()
