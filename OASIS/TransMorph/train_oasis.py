#!/usr/bin/env python3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import argparse
import glob
import csv
import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import neurite as ne
import matplotlib.pyplot as plt
import voxelmorph as vxm

from data import datasets, trans
import utils
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph


import matplotlib.pyplot as plt


class TransMorphAdapter(nn.Module):
    """Expose the repository TransMorph model through the reference trainer API."""

    def __init__(self, config_name='TransMorph-Large', inshape=None):
        super().__init__()
        config = copy.deepcopy(CONFIGS_TM[config_name])
        if inshape is not None:
            config.img_size = tuple(inshape)
        self.model = TransMorph.TransMorph(config)
        self.inshape = tuple(config.img_size)
        self.ndim = 3
        self.use_pyramid = False
        self.use_pdaps = False
        self.use_residual_flow_pyramid = False
        self.integrate = None
        self.spatial_transform = self.model.spatial_trans

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        source,
        target,
        return_warped_source=True,
        return_warped_target=False,
        return_field_type='displacement',
        return_coarse_flows=False,
        return_residual_flows=False,
        return_feature_edge_loss=False,
        feature_edge_indices=(0, 1),
    ):
        if return_field_type != 'displacement':
            raise ValueError('TransMorphAdapter only supports displacement fields.')
        if return_warped_target:
            raise ValueError('TransMorphAdapter does not support return_warped_target.')

        warped_source, displacement = self.model(torch.cat((source, target), dim=1))
        outputs = [displacement]
        if return_warped_source:
            outputs.append(warped_source)
        if return_coarse_flows:
            outputs.append([])
        if return_residual_flows:
            outputs.append([])
        if return_feature_edge_loss:
            outputs.append(displacement.new_tensor(0.0))
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


def build_registration_model(args, device):
    return TransMorphAdapter(
        config_name=args.transmorph_config,
        inshape=(160, 192, 224),
    ).to(device)

def save_qualitative_results(
    model,
    dataset,
    output_dir,
    epoch,
    device='cuda',
    suffix='',
    best_sample_idx=None,
    boundary_kernel='sobel',
    boundary_smooth_kernel=3,
    boundary_ring_inner_kernel=3,
    boundary_ring_outer_kernel=7,
):
    """Save mid-slice images of samples."""

    def _extract_error_guided_stage_slices(error_guided_maps, z_idx, reference_depth):
        stage_slices = []
        for item in error_guided_maps:
            error_map = item.get('error_map')
            gate_map = item.get('gate_map')
            if error_map is None or gate_map is None:
                continue
            map_depth = int(error_map.shape[4])
            if map_depth <= 1 or reference_depth <= 1:
                mapped_z_idx = 0
            else:
                mapped_z_idx = int(round(z_idx * (map_depth - 1) / (reference_depth - 1)))
            mapped_z_idx = max(0, min(mapped_z_idx, map_depth - 1))

            error_slice = np.rot90(error_map[0, 0, :, :, mapped_z_idx].detach().cpu().numpy(), -1)
            gate_slice = np.rot90(gate_map[0, 0, :, :, mapped_z_idx].detach().cpu().numpy(), -1)
            stage_slices.append({
                'stage_idx': item.get('stage_idx', -1),
                'error_slice': error_slice,
                'gate_slice': gate_slice,
            })
        stage_slices.sort(key=lambda x: x['stage_idx'])
        return stage_slices
    
    samples_to_plot = []
    
    # Get the dataset from the dataloader if needed, but we'll adapt to just take a list of data items
    # For OASIS dataset, data is (x, y, x_seg, y_seg)
    
    # Use medium validation sample for visualization: Index 9 (Initial DSC ~ 0.59)
    sample_default = dataset[9]
    samples_to_plot.append(('default', sample_default))
        
    for name_tag, sample in samples_to_plot:
        # Sample contains: x, y, x_seg, y_seg
        source = sample[0].unsqueeze(0).to(device)
        target = sample[1].unsqueeze(0).to(device)
        source_label = sample[2].unsqueeze(0).to(device)
        target_label = sample[3].unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(source, target, return_warped_source=True, return_field_type='displacement')
            displacement, warped_source = out[0], out[1]
            error_guided_maps = list(getattr(model, 'latest_error_guided_residual_maps', []))
            boundary_extractor = vxm.nn.modules.FixedGradientMagnitude3D(
                operator=boundary_kernel,
                smooth_kernel_size=boundary_smooth_kernel,
                normalize=True,
            ).to(device)
            source_boundary = boundary_extractor(source.float())
            target_boundary = boundary_extractor(target.float())
            warped_boundary = boundary_extractor(warped_source.float())
            base_fg_mask = build_foreground_mask(target.float(), target_label.float())
            boundary_ring_mask = build_boundary_ring_mask(
                base_fg_mask,
                inner_kernel_size=boundary_ring_inner_kernel,
                outer_kernel_size=boundary_ring_outer_kernel,
            )
            
            warped_label = None
            if source_label is not None:
                 trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                 warped_label = trf(source_label.float(), displacement)
        
        # Determine the best slice index
        # Default: middle slice
        slice_idx = source.shape[4] // 2
        
        # Extract slices using the determined index
        def get_slice(img_tensor, z_idx, is_label=False):
            if img_tensor is None: return None
            # img_tensor: (1, 1, X, Y, Z) (or similar)
            slice_tensor = img_tensor[:, 0, :, :, z_idx] if len(img_tensor.shape) == 5 else img_tensor[:, :, :, z_idx]
            # Remove batch dims
            slice_np = slice_tensor.detach().cpu().numpy()[0]
            if len(slice_np.shape) == 3: # if channel dim still exists
                slice_np = slice_np[0]
                
            if is_label:
                # Filter labels to only include: 
                # Lateral Ventricles (3, 22)
                # 3rd Ventricle (11), Thalamus (7, 26), Hippocampus (14, 30)
                labels_to_keep = [3, 7, 11, 14, 22, 26, 30]
                mask = np.isin(slice_np, labels_to_keep)
                slice_np = np.where(mask, slice_np, 0)
                
            # X-Y plane is Axial
            return np.rot90(slice_np, -1)

        src_slice = get_slice(source, slice_idx)
        tgt_slice = get_slice(target, slice_idx)
        warped_slice = get_slice(warped_source, slice_idx)
        src_boundary_slice = get_slice(source_boundary, slice_idx)
        tgt_boundary_slice = get_slice(target_boundary, slice_idx)
        warped_boundary_slice = get_slice(warped_boundary, slice_idx)
        ring_mask_slice = get_slice(boundary_ring_mask, slice_idx)
        
        src_lbl_slice = get_slice(source_label, slice_idx, is_label=True)
        tgt_lbl_slice = get_slice(target_label, slice_idx, is_label=True)
        warped_lbl_slice = get_slice(warped_label, slice_idx, is_label=True)
        
        has_labels = (src_lbl_slice is not None) and (tgt_lbl_slice is not None)
        error_guided_stage_slices = _extract_error_guided_stage_slices(error_guided_maps, slice_idx, source.shape[4])
        
        rows = 5 if len(error_guided_stage_slices) > 0 else 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        
        for ax in axes.flatten():
            ax.axis('off')
            
        # Determine global min and max for consistent brightness plotting
        vmax_val = max(np.max(src_slice), np.max(tgt_slice), np.max(warped_slice))
        vmin_val = min(np.min(src_slice), np.min(tgt_slice), np.min(warped_slice))
        
        # --- Row 1: Images & Differences ---
        
        # [0,0] Source Image
        axes[0, 0].imshow(src_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[0, 0].set_title('Source Image')
        axes[0, 0].axis('off')

        # [0,1] Target Image
        axes[0, 1].imshow(tgt_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')

        # [0,2] Diff: Moving - Fixed
        diff_moving_fixed = src_slice - tgt_slice
        im_diff1 = axes[0, 2].imshow(diff_moving_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 2].set_title('Diff: Source - Target')
        axes[0, 2].axis('off')

        # [0,3] Diff: Deformed - Target
        diff_warp_fixed = warped_slice - tgt_slice
        im_diff2 = axes[0, 3].imshow(diff_warp_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 3].set_title('Diff: Deformed - Target')
        axes[0, 3].axis('off')
        
        # --- Row 2: Label Overlays & Result vs GT ---
        if has_labels:
            def get_color(lbl, is_fixed=False):
                # Group Left/Right variants of the same structure to the same color
                lbl_map = {3: 0, 22: 0, 7: 1, 26: 1, 11: 2, 14: 3, 30: 3}
                mapped_lbl = lbl_map.get(int(lbl), int(lbl))
                
                # Hand-picked colors to ensure high visibility on gray images
                # Format: (Source/Moving color, Target/Fixed color)
                # Ensure they are same color family but explicitly distinguishable and both bright
                color_pairs = {
                    0: ('#1f77b4', '#00bfff'), # Dark Blue vs Deep Sky Blue
                    1: ('#ff7f0e', '#ffd700'), # Orange vs Gold/Yellow
                    2: ('#2ca02c', '#32cd32'), # Green vs Lime Green
                    3: ('#d62728', '#ff69b4'), # Red vs Hot Pink
                }
                
                if mapped_lbl in color_pairs:
                    hex_color = color_pairs[mapped_lbl][1 if is_fixed else 0]
                else:
                    hex_color = '#ffffff'
                    
                import matplotlib.colors as mcolors
                return mcolors.to_rgba(hex_color)

            def plot_label_contour(ax, bg_img, label_img, title, is_fixed=False):
                ax.imshow(bg_img, cmap='gray', vmin=vmin_val, vmax=vmax_val)
                if label_img is not None:
                     unique_labels = np.unique(label_img)
                     unique_labels = unique_labels[unique_labels > 0]
                     
                     for lbl in unique_labels:
                         mask = (label_img == lbl)
                         c = get_color(lbl, is_fixed=is_fixed)
                         if np.any(mask):
                             ax.contour(mask, colors=[c], linewidths=1.2)
                ax.set_title(title)
                ax.axis('off')

            plot_label_contour(axes[1, 0], src_slice, src_lbl_slice, 'Source + Labels', is_fixed=False)
            plot_label_contour(axes[1, 1], tgt_slice, tgt_lbl_slice, 'Target + Labels', is_fixed=True)
            plot_label_contour(axes[1, 2], warped_slice, warped_lbl_slice, 'Deformed + Labels', is_fixed=False)
            
        # [1,3] Overlay Warped Label on Target Image
        axes[1, 3].imshow(warped_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        if has_labels:
             unique_labels = np.unique(np.concatenate([tgt_lbl_slice, warped_lbl_slice]))
             unique_labels = unique_labels[unique_labels > 0]
             
             for lbl in unique_labels:
                 mask_tgt = (tgt_lbl_slice == lbl)
                 c_fixed = get_color(lbl, is_fixed=True)
                 if np.any(mask_tgt):
                     axes[1, 3].contour(mask_tgt, colors=[c_fixed], linewidths=1.5, linestyles='dashed', alpha=0.8)
                     
                 mask_warp = (warped_lbl_slice == lbl)
                 c_moving = get_color(lbl, is_fixed=False)
                 if np.any(mask_warp):
                     axes[1, 3].contour(mask_warp, colors=[c_moving], linewidths=1.5, linestyles='solid')

        axes[1, 3].set_title('Result vs GT')
        axes[1, 3].axis('off')

        # --- Row 3: Flow, Grid & Jacobian ---
        
        H, W = src_slice.shape
        grid_spacing = 10
        raw_d_slice = displacement.detach().cpu().numpy()[0, :, :, :, slice_idx] 
        d_slice = np.rot90(raw_d_slice, -1, axes=(1, 2))
        d_slice[1] = -d_slice[1] # flip Y displacement to match 180 deg rotation
        d_slice[2] = -d_slice[2] # flip X displacement to match 180 deg rotation

        dx, dy, dz = d_slice[2], d_slice[1], d_slice[0]
        max_mag = np.max(np.abs(d_slice)) + 1e-5
        
        flow_vis = np.zeros((H, W, 3), dtype=np.float32)
        flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5
        flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5
        flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5
        flow_vis = np.clip(flow_vis, 0, 1)
        
        # [2,0] Flow
        axes[2, 0].imshow(flow_vis)
        axes[2, 0].set_title('RGB Displacement')
        axes[2, 0].axis('off')

        from matplotlib.colors import hsv_to_rgb
        
        # [2,1] 3D Vector Legend
        ax_legend_spot = axes[2, 1]
        ax_legend_spot.clear()
        ax_legend_spot.axis('off')
        ax_legend_spot.set_aspect('equal')
        ax_legend_spot.set_xlim(-1.2, 1.2)
        ax_legend_spot.set_ylim(-1.2, 1.2)

        x_wheel = np.linspace(-0.12, 0.12, 100) 
        y_wheel = np.linspace(-0.12, 0.12, 100)
        XW, YW = np.meshgrid(x_wheel, y_wheel)
        RW = np.sqrt(XW**2 + YW**2)
        TW = np.arctan2(YW, XW)
        TW[TW < 0] += 2*np.pi
        
        HW = TW / (2*np.pi)
        SW = np.ones_like(HW)
        VW = np.ones_like(HW)
        mask = (RW <= 0.12)
        
        HSV_W = np.stack((HW, SW, VW), axis=-1)
        RGB_W = hsv_to_rgb(HSV_W)
        RGBA_W = np.concatenate([RGB_W, mask[..., None].astype(float)], axis=-1)
        
        ax_legend_spot.imshow(RGBA_W, extent=[-0.12, 0.12, -0.12, 0.12], origin='lower')

        o_x, o_y = 0, 0
        vec_x = np.array([0.5, -0.2])
        vec_y = np.array([-0.4, -0.25])
        vec_z = np.array([0.0, 0.5])
        
        scale = 0.312
        
        ax_legend_spot.arrow(o_x, o_y, vec_x[0]*scale, vec_x[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_y[0]*scale, vec_y[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_z[0]*scale, vec_z[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        
        ax_legend_spot.text(vec_x[0]*scale*1.6, vec_x[1]*scale*1.6, 'x', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_y[0]*scale*1.6, vec_y[1]*scale*1.6, 'y', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_z[0]*scale*1.6, vec_z[1]*scale*1.4, 'z', fontweight='bold', fontsize=16, ha='center', va='bottom')
        
        range_text = f"[{ -max_mag:.2f}, {max_mag:.2f}]"
        ax_legend_spot.text(0, -0.35, range_text, ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        # [2,2] Deformed Grid
        axes[2, 2].imshow(np.zeros_like(src_slice), cmap='gray', vmin=0, vmax=1) 
        # Plot vertical lines
        for i in range(0, W, grid_spacing):
            if i < d_slice.shape[2]:
                x_plot = i + d_slice[2, :, i]
                y_plot = np.arange(H) + d_slice[1, :, i]
                axes[2, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        # Plot horizontal lines
        for j in range(0, H, grid_spacing):
            if j < d_slice.shape[1]:
                x_plot = np.arange(W) + d_slice[2, j, :]
                y_plot = j + d_slice[1, j, :]
                axes[2, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        axes[2, 2].set_title('Deformed Grid')
        axes[2, 2].set_ylim(H, 0)
        axes[2, 2].set_xlim(0, W)
        axes[2, 2].axis('off')

        # [2,3] Jacobian Determinant
        disp_np = displacement.detach().cpu().numpy()[0]
        dz_dz, dz_dy, dz_dx = np.gradient(disp_np[0])
        dy_dz, dy_dy, dy_dx = np.gradient(disp_np[1])
        dx_dz, dx_dy, dx_dx = np.gradient(disp_np[2])
        
        jac_det = ( (1 + dx_dx) * ((1 + dy_dy) * (1 + dz_dz) - dy_dz * dz_dy)
                  - dx_dy * (dy_dx * (1 + dz_dz) - dy_dz * dz_dx)
                  + dx_dz * (dy_dx * dz_dy - (1 + dy_dy) * dz_dx) )
                  
        jac_slice = jac_det[:, :, slice_idx]
        jac_slice = np.rot90(jac_slice, -1)
        
        jac_vis = np.zeros((H, W, 3), dtype=np.float32)
        color_red = np.array([1.0, 0.0, 0.0])
        color_green = np.array([0.4, 0.8, 0.4]) 
        color_blue = np.array([0.4, 0.6, 0.9])  
        
        jac_vis[jac_slice < 0] = color_red
        jac_vis[(jac_slice >= 0) & (jac_slice <= 1)] = color_green
        jac_vis[jac_slice > 1] = color_blue
        
        axes[2, 3].imshow(jac_vis)
        axes[2, 3].set_title('Jacobian Determinant')
        axes[2, 3].axis('off')

        # --- Row 4: Boundary Maps & Ring Mask ---
        boundary_vmax = max(
            np.max(src_boundary_slice),
            np.max(tgt_boundary_slice),
            np.max(warped_boundary_slice),
            1e-6,
        )

        axes[3, 0].imshow(src_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 0].set_title('Source Boundary Map')
        axes[3, 0].axis('off')

        axes[3, 1].imshow(tgt_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 1].set_title('Target Boundary Map')
        axes[3, 1].axis('off')

        axes[3, 2].imshow(warped_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 2].set_title('Warped Boundary Map')
        axes[3, 2].axis('off')

        axes[3, 3].imshow(tgt_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[3, 3].imshow(ring_mask_slice, cmap='autumn', alpha=0.45, vmin=0.0, vmax=1.0)
        axes[3, 3].set_title('Boundary Ring Mask')
        axes[3, 3].axis('off')

        if len(error_guided_stage_slices) > 0:
            for plot_idx in range(4):
                axes[4, plot_idx].axis('off')

            for stage_pos, stage_item in enumerate(error_guided_stage_slices[:2]):
                col_offset = stage_pos * 2
                error_slice = stage_item['error_slice']
                gate_slice = stage_item['gate_slice']
                stage_idx = stage_item['stage_idx']

                error_vmax = max(float(np.max(error_slice)), 1e-6)
                axes[4, col_offset].imshow(error_slice, cmap='magma', vmin=0.0, vmax=error_vmax)
                axes[4, col_offset].set_title(f'Err Map (Stage {stage_idx})')
                axes[4, col_offset].axis('off')

                axes[4, col_offset + 1].imshow(gate_slice, cmap='viridis', vmin=0.5, vmax=1.5)
                axes[4, col_offset + 1].set_title(f'Gate Map (Stage {stage_idx})')
                axes[4, col_offset + 1].axis('off')

        plt.suptitle(f'Epoch {epoch} - Sample {name_tag} (Slice Z={slice_idx})', fontsize=16)
        
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except UserWarning:
            pass
        
        safe_tag = str(name_tag).replace('/', '_').replace('\\', '_')
        filename_out = f'vis_epoch_{epoch:04d}{suffix}_{safe_tag}.png'
        out_file = output_dir / filename_out
        plt.savefig(str(out_file))
        plt.close(fig)

import scipy.ndimage

def compute_hd95(ground_truth, prediction, spacing=None):
    if ground_truth.sum() == 0 or prediction.sum() == 0:
        return np.nan
    pred_border = prediction ^ scipy.ndimage.binary_erosion(prediction)
    gt_border = ground_truth ^ scipy.ndimage.binary_erosion(ground_truth)
    pts_pred = np.argwhere(pred_border)
    pts_gt = np.argwhere(gt_border)
    if pts_pred.shape[0] == 0 or pts_gt.shape[0] == 0:
        return np.nan
    if spacing is not None:
        pts_pred = pts_pred * np.array(spacing)
        pts_gt = pts_gt * np.array(spacing)
    from scipy.spatial import cKDTree
    kd_tree_gt = cKDTree(pts_gt)
    distances_pred_to_gt, _ = kd_tree_gt.query(pts_pred)
    kd_tree_pred = cKDTree(pts_pred)
    distances_gt_to_pred, _ = kd_tree_pred.query(pts_gt)
    return max(np.percentile(distances_pred_to_gt, 95), np.percentile(distances_gt_to_pred, 95))

def validate(

    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    compute_extra: bool = False,
    image_loss_fn: nn.Module = None,
    grad_loss_fn: nn.Module = None,
    loss_weights: list = None,
    image_loss_fn_coarse: nn.Module = None,
    boundary_loss_fn: nn.Module = None,
    boundary_loss_weight: float = 0.0,
    boundary_ring_inner_kernel: int = 3,
    boundary_ring_outer_kernel: int = 7,
    sasc_loss_fn: nn.Module = None,
    sasc_loss_weight: float = 0.0,
    feature_edge_loss_weight: float = 0.0,
    feature_edge_indices: tuple = (0, 1),
    pyramid_weight: float = 0.5,
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
    use_mask: bool = False,
    loss_type: str = 'mse',
) -> tuple:
    model.eval()
    eval_loss = utils.AverageMeter()
    eval_boundary_loss = utils.AverageMeter()
    eval_dsc = utils.AverageMeter()
    eval_hd95 = utils.AverageMeter()
    eval_jac = utils.AverageMeter()
    eval_mag = utils.AverageMeter()

    # Spatial transformation for nearest neighbour
    reg_model = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)

    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].to(device)
            x_seg = data[2].to(device)
            y_seg = data[3].to(device)

            # Get the displacement fields
            use_feature_edge_loss = feature_edge_loss_weight > 0 and hasattr(model, '_feature_edge_loss')
            use_residual_flow_pyramid = getattr(model, 'use_residual_flow_pyramid', False)
            supports_coarse_flows = getattr(model, 'use_pdaps', False) or use_residual_flow_pyramid
            request_pyramid_outputs = supports_coarse_flows and (pyramid_weight > 0 or pyramid_image_weight > 0 or residual_flow_reg_weight > 0)
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=request_pyramid_outputs,
                return_residual_flows=use_residual_flow_pyramid and request_pyramid_outputs,
                return_feature_edge_loss=use_feature_edge_loss,
                feature_edge_indices=feature_edge_indices,
            )
            displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss = unpack_model_outputs(
                out,
                expect_coarse_flows=request_pyramid_outputs,
                expect_residual_flows=use_residual_flow_pyramid and request_pyramid_outputs,
                expect_feature_edge_loss=use_feature_edge_loss,
            )

            target_float = y.float()
            warped_float = warped_source.float()
            base_fg_mask = build_foreground_mask(target_float, y_seg)
            boundary_ring_mask = build_boundary_ring_mask(
                base_fg_mask,
                inner_kernel_size=boundary_ring_inner_kernel,
                outer_kernel_size=boundary_ring_outer_kernel,
            )
            if use_mask:
                fg_mask = base_fg_mask
            else:
                fg_mask = torch.ones_like(target_float)

            boundary_loss = displacement.new_tensor(0.0)
            sasc_loss = displacement.new_tensor(0.0)
            if image_loss_fn is not None and grad_loss_fn is not None and loss_weights is not None:
                if loss_type == 'mse':
                    squared_diff = (target_float - warped_float) ** 2
                    img_loss = (squared_diff * fg_mask).sum() / (fg_mask.sum() + 1e-8)
                elif loss_type == 'ncc':
                    if use_mask:
                        img_loss = -image_loss_fn(target_float * fg_mask, warped_float * fg_mask).mean()
                    else:
                        img_loss = -image_loss_fn(target_float, warped_float).mean()
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                grad_loss = grad_loss_fn(displacement.float()).mean()
                loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss

                deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss = compute_oasis_pyramid_losses(
                    model,
                    x.float(),
                    target_float,
                    fg_mask,
                    coarse_flows,
                    residual_flows,
                    image_loss_fn_coarse,
                    grad_loss_fn,
                    loss_weights,
                    use_mask,
                    loss_type,
                    device=device,
                    pyramid_weight=pyramid_weight,
                    pyramid_image_weight=pyramid_image_weight,
                    pyramid_image_weights=pyramid_image_weights,
                    residual_flow_reg_weight=residual_flow_reg_weight,
                )
                if deep_sup_loss.requires_grad or deep_sup_loss.item() != 0:
                    loss = loss + deep_sup_loss
                if pyramid_image_weight > 0:
                    loss = loss + pyramid_image_weight * pyramid_img_loss

                if boundary_loss_fn is not None and boundary_loss_weight > 0:
                    boundary_loss = boundary_loss_fn(warped_float, target_float, mask=boundary_ring_mask)
                    loss = loss + boundary_loss_weight * boundary_loss

                if sasc_loss_fn is not None and sasc_loss_weight > 0:
                    sasc_loss = sasc_loss_fn(warped_float, target_float, mask=base_fg_mask)
                    loss = loss + sasc_loss_weight * sasc_loss

                if feature_edge_loss_weight > 0:
                    loss = loss + feature_edge_loss_weight * feature_edge_loss.float()

                if residual_flow_reg_weight > 0:
                    loss = loss + residual_flow_reg_weight * residual_flow_reg_loss

                eval_loss.update(loss.item(), x.size(0))
                eval_boundary_loss.update(boundary_loss.item(), x.size(0))

            # Warp the segmentations with nearest neighbour
            def_out = reg_model(x_seg.float(), displacement)

            # DSC
            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            eval_dsc.update(dsc.item(), x.size(0))

            if compute_extra:
                # Magnitude
                disp_mag = torch.sqrt(torch.sum(displacement ** 2, dim=1))
                eval_mag.update(disp_mag.mean().item(), x.size(0))

                # Jacobian
                disp_np = displacement.cpu().numpy()
                disp_np = np.transpose(disp_np, (0, 2, 3, 4, 1))
                target_np = y.cpu().numpy()
                batch_neg_jac = 0.0
                for i in range(disp_np.shape[0]):
                    jac_det = vxm.py.utils.jacobian_determinant(disp_np[i])
                    mask = target_np[i, 0] > 0.01
                    if jac_det.shape != mask.shape:
                        diff = np.array(mask.shape) - np.array(jac_det.shape)
                        ds, hs, ws = diff // 2
                        de, he, we = mask.shape[0] - (diff[0]-ds), mask.shape[1] - (diff[1]-hs), mask.shape[2] - (diff[2]-ws)
                        mask = mask[ds:de, hs:he, ws:we]
                    valid_sum = np.sum(mask)
                    if valid_sum > 0:
                        batch_neg_jac += np.sum((jac_det <= 0) & mask) / valid_sum
                eval_jac.update(batch_neg_jac / disp_np.shape[0], x.size(0))

                # HD95
                wl_np = def_out.cpu().numpy()
                tl_np = y_seg.cpu().numpy()
                batch_hd95_sum = 0.0
                batch_hd95_count = 0
                for b in range(wl_np.shape[0]):
                    u_labels = np.unique(np.concatenate((wl_np[b], tl_np[b])))
                    u_labels = u_labels[u_labels > 0.5]
                    for l in u_labels:
                        mask_pred = (wl_np[b, 0] == l)
                        mask_gt = (tl_np[b, 0] == l)
                        hd = compute_hd95(mask_gt, mask_pred)
                        if not np.isnan(hd):
                            batch_hd95_sum += hd
                            batch_hd95_count += 1
                if batch_hd95_count > 0:
                    eval_hd95.update(batch_hd95_sum / batch_hd95_count, x.size(0))

    if compute_extra:
        return eval_loss.avg, eval_boundary_loss.avg, eval_dsc.avg, eval_hd95.avg, eval_jac.avg, eval_mag.avg
    return eval_loss.avg, eval_boundary_loss.avg, eval_dsc.avg

def _binary_dilate_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def _binary_erode_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def build_foreground_mask(target: torch.Tensor, target_seg: torch.Tensor = None) -> torch.Tensor:
    if target_seg is not None and (target_seg > 0).float().sum() >= 1e-3:
        mask = (target_seg > 0).float()
    else:
        mask = (target > 0.01).float()

    mask = _binary_dilate_3d(mask, kernel_size=5)
    mask = _binary_erode_3d(mask, kernel_size=5)
    mask = _binary_dilate_3d(mask, kernel_size=3)
    return mask.clamp_(0.0, 1.0)

def build_boundary_ring_mask(
    foreground_mask: torch.Tensor,
    inner_kernel_size: int = 3,
    outer_kernel_size: int = 7,
) -> torch.Tensor:
    inner_kernel_size = max(1, int(inner_kernel_size))
    outer_kernel_size = max(inner_kernel_size, int(outer_kernel_size))

    if inner_kernel_size % 2 == 0:
        inner_kernel_size += 1
    if outer_kernel_size % 2 == 0:
        outer_kernel_size += 1

    foreground_mask = foreground_mask.float().clamp_(0.0, 1.0)
    outer_band = _binary_dilate_3d(foreground_mask, kernel_size=outer_kernel_size)
    inner_core = _binary_erode_3d(foreground_mask, kernel_size=inner_kernel_size)
    ring_mask = (outer_band - inner_core).clamp_(0.0, 1.0)

    if torch.count_nonzero(ring_mask).item() == 0:
        return foreground_mask
    return ring_mask

def parse_feature_edge_indices(scales: str) -> tuple:
    scale_to_index = {'1/2': 0, '1/4': 1, '1/8': 2, '1/16': 3}
    indices = []
    for token in scales.split(','):
        token = token.strip()
        if not token:
            continue
        if token in scale_to_index:
            indices.append(scale_to_index[token])
        else:
            indices.append(int(token))
    return tuple(indices)


def parse_pyramid_image_weights(weights: str) -> tuple:
    values = []
    for token in weights.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return tuple(values)


def parse_sasc_scales(scales: str) -> tuple:
    values = []
    for token in scales.split(','):
        token = token.strip()
        if not token:
            continue
        if '/' in token:
            numerator, denominator = token.split('/', 1)
            values.append(float(numerator) / float(denominator))
        else:
            values.append(float(token))
    return tuple(values)


def _match_weight_count(weights: tuple, count: int) -> list:
    if count <= 0:
        return []
    if not weights:
        return [1.0] * count
    matched = list(weights[:count])
    if len(matched) < count:
        matched.extend([matched[-1]] * (count - len(matched)))
    return matched


def unpack_model_outputs(
    out,
    expect_coarse_flows: bool = False,
    expect_residual_flows: bool = False,
    expect_feature_edge_loss: bool = False,
):
    idx = 0
    displacement = out[idx]
    idx += 1
    warped_source = out[idx] if len(out) > idx else None
    idx += 1

    coarse_flows = []
    if expect_coarse_flows:
        coarse_flows = out[idx]
        idx += 1

    residual_flows = []
    if expect_residual_flows:
        residual_flows = out[idx]
        idx += 1

    feature_edge_loss = displacement.new_tensor(0.0)
    if expect_feature_edge_loss and len(out) > idx:
        feature_edge_loss = out[idx]

    return displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss


def global_ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    u_true = y_true.mean(dim=[1, 2, 3, 4], keepdim=True)
    u_pred = y_pred.mean(dim=[1, 2, 3, 4], keepdim=True)
    z_true = y_true - u_true
    z_pred = y_pred - u_pred
    cov = (z_true * z_pred).mean(dim=[1, 2, 3, 4])
    var_true = (z_true ** 2).mean(dim=[1, 2, 3, 4])
    var_pred = (z_pred ** 2).mean(dim=[1, 2, 3, 4])
    corr = cov / (torch.sqrt(var_true * var_pred) + eps)
    return -corr.mean()


def compute_oasis_pyramid_losses(
    model: nn.Module,
    source_float: torch.Tensor,
    target_float: torch.Tensor,
    fg_mask: torch.Tensor,
    coarse_flows: list,
    residual_flows: list,
    image_loss_fn_coarse: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: list,
    use_mask: bool,
    loss_type: str,
    device: str,
    pyramid_weight: float = 0.5,
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
):
    deep_sup_loss = target_float.new_tensor(0.0)
    pyramid_img_loss = target_float.new_tensor(0.0)
    residual_flow_reg_loss = target_float.new_tensor(0.0)

    if len(coarse_flows) > 0:
        scale_weights = _match_weight_count(pyramid_image_weights, len(coarse_flows))
        image_weight_sum = max(sum(scale_weights), 1e-6)
        st_cache = {}
        for scale_weight, c_flow in zip(scale_weights, coarse_flows):
            c_shape = c_flow.shape[2:]
            c_flow_float = c_flow.float()
            if pyramid_weight > 0:
                c_grad_loss = grad_loss_fn(c_flow_float).mean()
                deep_sup_loss = deep_sup_loss + c_grad_loss

            if pyramid_image_weight > 0:
                if target_float.shape[2:] != c_shape:
                    y_down = F.interpolate(target_float, size=c_shape, mode='trilinear', align_corners=False)
                    x_down = F.interpolate(source_float, size=c_shape, mode='trilinear', align_corners=False)
                    mask_down = F.interpolate(fg_mask, size=c_shape, mode='trilinear', align_corners=False) if use_mask else None
                else:
                    y_down = target_float
                    x_down = source_float
                    mask_down = fg_mask if use_mask else None

                if hasattr(model, 'integrate') and model.integrate is not None:
                    c_disp = model.integrate(c_flow_float)
                else:
                    c_disp = c_flow_float

                if c_shape not in st_cache:
                    st_cache[c_shape] = vxm.nn.modules.SpatialTransformer().to(device)

                warped_x_down = st_cache[c_shape](x_down, c_disp)

                if loss_type == 'mse':
                    squared_diff = (y_down - warped_x_down) ** 2
                    if use_mask and mask_down is not None:
                        c_img_loss = (squared_diff * mask_down).sum() / (mask_down.sum() + 1e-8)
                    else:
                        c_img_loss = squared_diff.mean()
                else:
                    if use_mask and mask_down is not None:
                        if image_loss_fn_coarse is not None:
                            c_img_loss = -image_loss_fn_coarse(y_down * mask_down, warped_x_down * mask_down).mean()
                        else:
                            c_img_loss = global_ncc(y_down * mask_down, warped_x_down * mask_down)
                    else:
                        if image_loss_fn_coarse is not None:
                            c_img_loss = -image_loss_fn_coarse(y_down, warped_x_down).mean()
                        else:
                            c_img_loss = global_ncc(y_down, warped_x_down)

                pyramid_img_loss = pyramid_img_loss + scale_weight * c_img_loss

        if pyramid_weight > 0:
            deep_sup_loss = (deep_sup_loss / len(coarse_flows)) * pyramid_weight
        if pyramid_image_weight > 0:
            pyramid_img_loss = pyramid_img_loss / image_weight_sum

    if residual_flow_reg_weight > 0 and len(residual_flows) > 0:
        stage_weights = [float(idx + 1) for idx in range(len(residual_flows))]
        weight_sum = max(sum(stage_weights), 1e-6)
        for stage_weight, residual_flow in zip(stage_weights, residual_flows):
            residual_flow_reg_loss = residual_flow_reg_loss + stage_weight * grad_loss_fn(residual_flow.float()).mean()
        residual_flow_reg_loss = residual_flow_reg_loss / weight_sum

    return deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: list,
    pyramid_weight: float = 0.5,
    device: str = 'cuda',
    scaler = None,
    amp_enabled: bool = True,
    use_mask: bool = False,
    loss_type: str = 'mse',
    image_loss_fn_coarse: nn.Module = None,
    boundary_loss_fn: nn.Module = None,
    boundary_loss_weight: float = 0.0,
    boundary_ring_inner_kernel: int = 3,
    boundary_ring_outer_kernel: int = 7,
    sasc_loss_fn: nn.Module = None,
    sasc_loss_weight: float = 0.0,
    feature_edge_loss_weight: float = 0.0,
    feature_edge_indices: tuple = (0, 1),
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_img_loss = 0.0
    total_grad_loss = 0.0
    total_sasc_loss = 0.0
    total_feature_edge_loss = 0.0
    valid_batches = 0

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        # TransMorph OASISDataset returns: x, y, x_seg, y_seg
        # x: moving image, y: fixed image
        x = data[0].to(device)
        y = data[1].to(device)
        # y_seg 常作为脑内部组织的金标准，适合当做高信度的前景色 Mask 候选
        y_seg = data[3].to(device)

        # 使用 AMP autocast
        with torch.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            use_feature_edge_loss = feature_edge_loss_weight > 0 and hasattr(model, '_feature_edge_loss')
            use_residual_flow_pyramid = getattr(model, 'use_residual_flow_pyramid', False)
            # Get the displacement and the warped source image from the model
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=True,
                return_residual_flows=use_residual_flow_pyramid,
                return_feature_edge_loss=use_feature_edge_loss,
                feature_edge_indices=feature_edge_indices,
            )
            displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss = unpack_model_outputs(
                out,
                expect_coarse_flows=True,
                expect_residual_flows=use_residual_flow_pyramid,
                expect_feature_edge_loss=use_feature_edge_loss,
            )

        # 🔥【关键修复】：在这里退出模型前向的 AMP autocast 作用域！
        # 如果把损失函数（不论是 NCC 还是 MSE）放在 autocast 里面算，因为 Loss 内部往往会有 Conv3d 或者平方项操作，
        # PyTorch 会非常聪明地强行把输入的 float32 降级成 BF16/FP16 再算，这会导致 NCC 的方差发生严重的精度截断甚至变成负数导致 NaN。
        # AMP 兼容性保护：强制将预测结果和 Loss 计算切回 float32
        target_float = y.float()
        warped_float = warped_source.float()
        
        # --- Mask 核心逻辑联动 ---
        # Q: 前景 Mask 的过滤规则如果是按 > 0.01 背景排查，图像内部会有 0 体素吗？
        # A: 会。脑室（CSF/脑脊液区域）、肿瘤病灶、或者是扫描伪影在某些 MRI 模态(如T1) 中，部分像素可能天然表现为黑(接近0)。
        # 由于简单的硬阈值 `> 0.01` 会不小心在内部抠出“空洞”，一般用下述 2 种方案处理：
        # 方法A (推荐): 如果你的数据集里自带真实器官的 Label `y_seg`，直接拿全器官 Label 生成 Mask `(y_seg > 0).float()` 即可完美覆盖目标实质区域！
        # 方法B (常规): 依然用阈值提取背景，但做膨胀/闭运算填补内部空洞(morphological hole filling)。但深度学习中往往算算算嫌麻烦。
        
        base_fg_mask = build_foreground_mask(target_float, y_seg)
        boundary_ring_mask = build_boundary_ring_mask(
            base_fg_mask,
            inner_kernel_size=boundary_ring_inner_kernel,
            outer_kernel_size=boundary_ring_outer_kernel,
        )

        if use_mask:
            fg_mask = base_fg_mask
        else:
            # 也就是默认在全图 (Batchx1xHxWxD) 上一视同仁全部计算 Loss
            fg_mask = torch.ones_like(target_float)
        
        if loss_type == 'mse':
            # 手动计算前景/全局的加权 MSE
            squared_diff = (target_float - warped_float) ** 2
            img_loss = (squared_diff * fg_mask).sum() / (fg_mask.sum() + 1e-8)
        elif loss_type == 'ncc':
            if use_mask:
                # 屏蔽掉非脑范围，强制外围全黑，中心有效，使得 NCC 计算更稳定且聚焦大脑
                masked_target = target_float * fg_mask
                masked_warped = warped_float * fg_mask
                img_loss = -image_loss_fn(masked_target, masked_warped).mean()
            else:
                img_loss = -image_loss_fn(target_float, warped_float).mean()
        
        grad_loss = grad_loss_fn(displacement.float()).mean()
        boundary_loss = displacement.new_tensor(0.0)
        if boundary_loss_fn is not None and boundary_loss_weight > 0:
            boundary_loss = boundary_loss_fn(warped_float, target_float, mask=boundary_ring_mask)
        sasc_loss = displacement.new_tensor(0.0)
        if sasc_loss_fn is not None and sasc_loss_weight > 0:
            sasc_loss = sasc_loss_fn(warped_float, target_float, mask=base_fg_mask)
        
        deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss = compute_oasis_pyramid_losses(
            model,
            x.float(),
            target_float,
            fg_mask,
            coarse_flows,
            residual_flows,
            image_loss_fn_coarse,
            grad_loss_fn,
            loss_weights,
            use_mask,
            loss_type,
            device=device,
            pyramid_weight=pyramid_weight,
            pyramid_image_weight=pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=residual_flow_reg_weight,
        )

        grad_loss = torch.clamp(grad_loss, min=0.0, max=100.0)
        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        if boundary_loss_weight > 0:
            loss = loss + boundary_loss_weight * boundary_loss
        if sasc_loss_weight > 0:
            loss = loss + sasc_loss_weight * sasc_loss
        if feature_edge_loss_weight > 0:
            loss = loss + feature_edge_loss_weight * feature_edge_loss.float()
        if deep_sup_loss.requires_grad or deep_sup_loss.item() != 0:
            loss = loss + deep_sup_loss # Add deep supervision component
        if pyramid_image_weight > 0:
            loss = loss + pyramid_image_weight * pyramid_img_loss
        if residual_flow_reg_weight > 0:
            loss = loss + residual_flow_reg_weight * residual_flow_reg_loss

        # 数值稳定性保护：发现非有限值则跳过该 batch，避免污染整轮 loss
        if not torch.isfinite(loss):
            print(
                f"[WARN] Non-finite loss at batch {batch_idx}: "
                f"img_loss={img_loss.item()}, grad_loss={grad_loss.item()}, boundary_loss={boundary_loss.item()}, sasc_loss={sasc_loss.item()}, feature_edge_loss={feature_edge_loss.item()}, total={loss.item()}"
            )
            # 彻底释放包含 NaN/Inf 计算图的所有局部变量，防止在遇到 NaN 直接 continue 时显存泄漏引发后续 OOM
            optimizer.zero_grad(set_to_none=True)
            del out, displacement, warped_source, coarse_flows, residual_flows, loss, img_loss, grad_loss, deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss
            torch.cuda.empty_cache()
            continue

        if amp_enabled and scaler is not None:
            # 缩放 loss，反向传播
            scaler.scale(loss).backward()
            # 在执行梯度裁剪前，必须先 unscale 梯度
            scaler.unscale_(optimizer)
            # 增加梯度裁剪，防止黑背景区导致的除零或梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item()
        total_img_loss += img_loss.item()
        total_grad_loss += grad_loss.item()
        total_sasc_loss += sasc_loss.item()
        total_feature_edge_loss += feature_edge_loss.float().item()
        valid_batches += 1

    if valid_batches == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    return (
        total_loss / valid_batches,
        total_img_loss / valid_batches,
        total_grad_loss / valid_batches,
        total_sasc_loss / valid_batches,
        total_feature_edge_loss / valid_batches,
    )

def main():
    parser = argparse.ArgumentParser(description='Train 3D TransMorph on OASIS data')
    parser.add_argument('--output', type=str, default='/root/autodl-tmp/models/oasis_transmorph.pt', help='Output model path')
    parser.add_argument('--transmorph-config', type=str, default='TransMorph-Large', choices=sorted(CONFIGS_TM), help='TransMorph model configuration')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ncc'], help='Image similarity loss')
    parser.add_argument('--use-mask', action='store_true', help='Use foreground mask to exclude black background in loss calculation')
    parser.add_argument('--disable-amp', action='store_true', help='Force disable AMP (Automatic Mixed Precision)')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01, help='Weight of gradient loss')
    parser.add_argument('--pyramid-weight', type=float, default=0.5, help='Weight for intermediate pyramid flow-smoothness deep supervision loss')
    parser.add_argument('--use-residual-flow-pyramid', action='store_true', help='Use lightweight coarse-to-fine residual flow accumulation without warping skip features')
    parser.add_argument('--use-error-guided-residual', action='store_true', help='Enhance residual flow pyramid with Structure-aware & Error-guided side branch')
    parser.add_argument("--error-guided-metric", type=str, default="feature_ncc", choices=["feature_ncc", "mind"], help="Metric to compute error maps for guidance")
    parser.add_argument('--residual-flow-limit', type=float, default=4.0, help='Per-stage magnitude cap for residual flow heads when residual flow pyramid is enabled')
    parser.add_argument('--pyramid-image-weight', type=float, default=0.0, help='Weight for multi-scale image supervision on intermediate pyramid displacements')
    parser.add_argument('--pyramid-image-weights', type=str, default='0.2,0.35,0.5', help='Comma-separated relative weights for intermediate pyramid image supervision from coarse to fine')
    parser.add_argument('--pyramid-image-start-epoch', type=int, default=1, help='Enable pyramid image supervision starting from this 1-based epoch')
    parser.add_argument('--residual-flow-reg-weight', type=float, default=0.0, help='Weight for gradient regularization on per-stage residual flow heads')
    parser.add_argument('--use-pdaps', action='store_true', help='Use Pyramid-guided Deformation-Aware Progressive Skip')
    parser.add_argument('--pdaps-flow-limit', type=float, default=20.0, help='Maximum physical flow limit for P-DAPS. E.g., 20.0 for Brain, 30.0+ for Abdomen CT/MRI.')
    parser.add_argument('--use-daps', action='store_true', help='Use original DAPS')
    parser.add_argument('--use-dsin', action='store_true', help='Enable DSIN in the shallow decoupled encoder layers')
    parser.add_argument('--use-cmim', action='store_true', help='Enable CMIM at deep decoder scales')
    parser.add_argument('--use-cross-mamba', action='store_true', help='Enable Cross-Mamba at deep decoder scales')
    parser.add_argument('--cross-mamba-scales', type=str, default='1/16,1/8', help='Comma-separated deep scales for Cross-Mamba, e.g. 1/16 or 1/16,1/8,1/4,1/2')
    parser.add_argument('--cross-mamba-offset-limit', type=float, default=0.0, help='Feature-space resampling offset cap inside Cross-Mamba; 0 disables offset limiting/resampling regularization for cleaner ablations')
    parser.add_argument('--cross-mamba-offset-smooth-kernel', type=int, default=1, help='Odd average-pooling kernel for smoothing Cross-Mamba offsets; 1 disables smoothing')
    parser.add_argument('--use-wcv', action='store_true', help='Enable window cross-attention on shallow skip features')
    parser.add_argument('--use-swcv', action='store_true', help='Enable Structure-Aware WCV on deep skip features')
    parser.add_argument('--use-gcv', action='store_true', help='Enable global cost volume on deep features')
    parser.add_argument('--use-boundary-branch', action='store_true', help='Enable lightweight boundary-guided feature modulation branch')
    parser.add_argument('--boundary-branch-scales', type=str, default='deep', choices=['deep', 'all'], help='Apply boundary branch on deep features only or all encoder scales')
    parser.add_argument('--boundary-branch-strength', type=float, default=0.5, help='Residual gate strength for the boundary feature branch')
    parser.add_argument('--use-boundary-loss', action='store_true', help='Enable explicit boundary consistency loss between warped source and target')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.1, help='Weight for boundary consistency loss when enabled')
    parser.add_argument('--boundary-loss-start-epoch', type=int, default=10, help='Enable boundary loss only after this 1-based epoch index; e.g. 10 starts boundary loss from epoch 11')
    parser.add_argument('--boundary-loss-metric', type=str, default='l1', choices=['ncc', 'l1'], help='Metric used by the boundary consistency loss; `l1` compares 3D Sobel gradient maps more directly and usually overlaps less with the main image NCC term')
    parser.add_argument('--boundary-kernel', type=str, default='sobel', choices=['sobel', 'diff'], help='Fixed operator used to extract 3D boundary maps')
    parser.add_argument('--boundary-smooth-kernel', type=int, default=3, help='Odd smoothing kernel size applied before boundary extraction')
    parser.add_argument('--boundary-ring-inner-kernel', type=int, default=3, help='Inner erosion kernel size for the foreground boundary ring mask')
    parser.add_argument('--boundary-ring-outer-kernel', type=int, default=7, help='Outer dilation kernel size for the foreground boundary ring mask')
    parser.add_argument('--use-sasc', action='store_true', help='Enable Self-supervised Anatomical Structure Consistency auxiliary loss')
    parser.add_argument('--sasc-weight', type=float, default=0.02, help='Total weight for SASC auxiliary loss')
    parser.add_argument('--sasc-start-epoch', type=int, default=20, help='Start SASC from this 1-based epoch')
    parser.add_argument('--sasc-ramp-epochs', type=int, default=10, help='Linearly ramp SASC weight over this many epochs after start; 0 disables ramp')
    parser.add_argument('--sasc-scales', type=str, default='1,1/2,1/4', help='Comma-separated SASC structure scales, e.g. 1,1/2,1/4')
    parser.add_argument('--sasc-boundary-weight', type=float, default=1.0, help='Internal boundary term weight inside SASC')
    parser.add_argument('--sasc-mind-weight', type=float, default=0.0, help='Internal MIND term weight inside SASC; usually 0 for single-modal OASIS')
    parser.add_argument('--sasc-mind-scale', type=float, default=0.25, help='Only compute SASC MIND at scales <= this value')
    parser.add_argument('--use-sdmr', action='store_true', help='Enable Structure-error-guided Diffeomorphic Mamba Refinement on the predicted velocity field')
    parser.add_argument('--sdmr-use-mind', action='store_true', help='Include low-resolution MIND residual maps in SDMR inputs')
    parser.add_argument('--sdmr-scale', type=float, default=0.125, help='Low-resolution scale used by SDMR refiner, e.g. 0.125 or 0.25')
    parser.add_argument('--sdmr-hidden-channels', type=int, default=16, help='Hidden channels of the SDMR refiner')
    parser.add_argument('--sdmr-flow-limit', type=float, default=1.0, help='Magnitude cap for SDMR residual velocity in full-resolution voxel units')
    parser.add_argument('--sdmr-alpha', type=float, default=0.5, help='Residual velocity blending weight for SDMR')
    parser.add_argument('--use-feature-edge-loss', action='store_true', help='Enable feature-level multi-scale gradient magnitude consistency on encoder features')
    parser.add_argument('--feature-edge-loss-weight', type=float, default=0.05, help='Weight for feature-level multi-scale gradient magnitude consistency loss')
    parser.add_argument('--feature-edge-scales', type=str, default='1/2,1/4', help='Comma-separated encoder scales or indices for feature edge loss, e.g. 1/2,1/4 or 0,1')
    parser.add_argument('--feature-edge-start-epoch', type=int, default=1, help='Enable feature-edge loss from this 1-based epoch index; e.g. 5 starts applying it at epoch 5')
    parser.add_argument('--encoder-type', type=str, default='cnn', choices=['cnn', 'mamba'], help='Backbone type for feature extraction.')
    parser.add_argument('--mamba-shallow-multi', action='store_true', help='Legacy shorthand: enable multi-axis (d,h,w) scanning for both encoder and decoder 1/8-scale Mamba blocks.')
    parser.add_argument('--mamba-enc-shallow-multi', action='store_true', help='Enable multi-axis (d,h,w) scanning for the encoder 1/8-scale Mamba block.')
    parser.add_argument('--mamba-dec-shallow-multi', action='store_true', help='Enable multi-axis (d,h,w) scanning for the decoder 1/8-scale Mamba block.')
    parser.add_argument('--mamba-quarter-scale', action='store_true', help='Enable Mamba scanning (single-axis) at 1/4 scale (i=1) to improve fine structural alignment.')
    parser.add_argument('--mamba-dec-quarter-scale', action='store_true', help='Enable decoder Mamba scanning (single-axis) at 1/4 scale (i=1) to improve fine structural alignment.')
    parser.add_argument('--mamba-parallel-block', action='store_true', help='Use Parallel Local(CNN)-Global(Mamba) Block instead of serial Mamba to protect local edges.')
    parser.add_argument('--decouple-layers', type=int, default=2, help='Number of shallow decoupled encoder layers used in dual-stream mode')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--window-size', type=int, default=9, help='Window size for WCV and S-WCV')
    parser.add_argument('--fusion-method', type=str, default='compress_concat', choices=['add', 'concat', 'compress_concat'], help='Feature fusion method for Siamese encoder')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--vis-every', type=int, default=5, help='Save qualitative visualization every N epochs; set 0 to disable')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument('--warm-start', type=int, default=10, help='Early stopping warm start steps')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of epochs for learning rate warmup')
    parser.add_argument('--integration-steps', type=int, default=0, help='number of integration steps for diffeomorphic registration')
    parser.add_argument('--train-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    args = parser.parse_args()

    unsupported_flags = (
        'use_residual_flow_pyramid', 'use_error_guided_residual', 'use_pdaps',
        'use_daps', 'use_dsin', 'use_cmim', 'use_cross_mamba', 'use_wcv',
        'use_swcv', 'use_gcv', 'use_boundary_branch', 'use_sdmr',
        'use_feature_edge_loss',
    )
    enabled_unsupported = [
        f'--{name.replace("_", "-")}' for name in unsupported_flags
        if getattr(args, name)
    ]
    if enabled_unsupported:
        parser.error(
            'The selected TransMorph architecture does not support: '
            + ', '.join(enabled_unsupported)
        )
    if args.integration_steps != 0:
        parser.error('TransMorph predicts displacement directly; --integration-steps must be 0.')

    if args.mamba_shallow_multi:
        args.mamba_enc_shallow_multi = True
        args.mamba_dec_shallow_multi = True

    if args.use_daps and args.use_pdaps:
        parser.error('--use-daps and --use-pdaps are mutually exclusive. Use --use-pdaps for the current pyramid design.')
    if args.use_cmim and args.use_cross_mamba:
        parser.error('--use-cmim and --use-cross-mamba are mutually exclusive interaction modules.')
    if args.use_error_guided_residual and not args.use_residual_flow_pyramid:
        parser.error('--use-error-guided-residual requires --use-residual-flow-pyramid.')
    if args.boundary_ring_inner_kernel < 1 or args.boundary_ring_outer_kernel < 1:
        parser.error('--boundary-ring-inner-kernel and --boundary-ring-outer-kernel must be positive integers.')
    if args.boundary_ring_outer_kernel < args.boundary_ring_inner_kernel:
        parser.error('--boundary-ring-outer-kernel must be greater than or equal to --boundary-ring-inner-kernel.')
    if args.feature_edge_start_epoch < 1:
        parser.error('--feature-edge-start-epoch must be a positive integer.')
    if args.sasc_start_epoch < 1:
        parser.error('--sasc-start-epoch must be a positive integer.')
    if args.sasc_ramp_epochs < 0:
        parser.error('--sasc-ramp-epochs must be non-negative.')
    if args.use_sasc and args.sasc_weight < 0:
        parser.error('--sasc-weight must be non-negative.')
    if args.use_sdmr and not (0 < args.sdmr_scale <= 1.0):
        parser.error('--sdmr-scale must be in (0, 1].')
    if args.use_sdmr and args.sdmr_hidden_channels < 1:
        parser.error('--sdmr-hidden-channels must be a positive integer.')

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 加速补丁：为固定尺寸的 3D 输入寻找最优 CUDA 卷积算子
    import torch.backends.cudnn as cudnn
    if device == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = False
        print("🚀 cuDNN Benchmark enabled for Cuda acceleration!")

    # Create model
    model = build_registration_model(args, device)
    print(f'TransMorph config: {args.transmorph_config}')

    # 统计并打印参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Total Trainable Parameters: {total_params:,}')

    # ==========================
    # AMP & Loss & Mask 核心联动策略
    # ==========================
    # 1. 损失函数策略
    if args.loss.lower() == 'ncc':
        # 增大 eps (默认是 1e-5)，防止由于图像大面积黑色背景(方差近乎 0)导致的除零/梯度爆炸
        image_loss_fn = ne.nn.modules.NCC(eps=1e-3)
        image_loss_fn_coarse = ne.nn.modules.NCC(eps=1e-2, window_size=3)
    else:
        image_loss_fn = ne.nn.modules.MSE()
        image_loss_fn_coarse = ne.nn.modules.MSE()
        
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    boundary_loss_fn = None
    if args.use_boundary_loss:
        boundary_loss_fn = vxm.nn.losses.BoundaryConsistencyLoss(
            operator=args.boundary_kernel,
            metric=args.boundary_loss_metric,
            smooth_kernel_size=args.boundary_smooth_kernel,
        ).to(device)
    sasc_loss_fn = None
    sasc_scales = parse_sasc_scales(args.sasc_scales)
    if args.use_sasc:
        sasc_loss_fn = vxm.nn.losses.SelfSupervisedAnatomicalStructureConsistencyLoss(
            scales=sasc_scales,
            boundary_weight=args.sasc_boundary_weight,
            mind_weight=args.sasc_mind_weight,
            mind_scale=args.sasc_mind_scale,
            operator=args.boundary_kernel,
            smooth_kernel_size=args.boundary_smooth_kernel,
            mind_radius=2,
            mind_dilation=2,
            mind_eps=1e-8,
        ).to(device)
    loss_weights = [1.0, args.lambda_param]
    feature_edge_indices = parse_feature_edge_indices(args.feature_edge_scales)
    pyramid_image_weights = parse_pyramid_image_weights(args.pyramid_image_weights)
    base_feature_edge_loss_weight = args.feature_edge_loss_weight if args.use_feature_edge_loss else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler: Warmup (Linear) then Cosine annealing to gradually lower LR
    warmup_epochs = args.warmup_epochs
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # 2. AMP 策略与防爆保护
    amp_enabled = (device == 'cuda') and not args.disable_amp
    if amp_enabled and (args.loss.lower() == 'ncc') and (args.integration_steps > 0):
        print("\n\n[INFO 🚀] 使用了 AMP + NCC。系统已修复精度范围域，请放心训练！\n\n")
        
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Dataloader identical to TransMorph
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    
    train_pattern = os.path.join(args.train_dir, '*.pkl')
    val_pattern = os.path.join(args.val_dir, '*.pkl')
    train_set = datasets.OASISBrainDataset(glob.glob(train_pattern), transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(glob.glob(val_pattern), transforms=val_composed)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    import datetime
    
    # Ensure output dir
    # Create a timestamped directory for this run to keep logs and checkpoints separate
    # Use Beijing Time (UTC+8)
    utc_now = datetime.datetime.utcnow()
    beijing_time = utc_now + datetime.timedelta(hours=8)
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    
    input_output_path = Path(args.output)

    # Structure: <parent>/<stem>_<timestamp>/<stem>.pt
    run_dir = input_output_path.parent / f"{input_output_path.stem}_{timestamp}"
    model_filename = f"{input_output_path.stem}{input_output_path.suffix}"
    out_path = run_dir / model_filename
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for this run: {out_path.parent}")

    # Logging setup
    log_file = out_path.parent / 'train_log.csv'
    
    # Save training configuration
    config_file = out_path.parent / 'config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model Architecture: {args.transmorph_config}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"Pyramid Weight: {args.pyramid_weight}\n")
        f.write(f"Pyramid Image Weight: {args.pyramid_image_weight}\n")
        f.write(f"Pyramid Image Start Epoch: {args.pyramid_image_start_epoch}\n")
        f.write(f"Pyramid Image Weights: {args.pyramid_image_weights}\n")
        f.write(f"Feature Edge Loss Weight: {base_feature_edge_loss_weight}\n")
        f.write(f"Feature Edge Start Epoch: {args.feature_edge_start_epoch}\n")
        f.write(f"Feature Edge Scales: {args.feature_edge_scales}\n")
        f.write(f"Use SASC: {args.use_sasc}\n")
        f.write(f"SASC Weight: {args.sasc_weight}\n")
        f.write(f"SASC Start Epoch: {args.sasc_start_epoch}\n")
        f.write(f"SASC Ramp Epochs: {args.sasc_ramp_epochs}\n")
        f.write(f"SASC Scales: {args.sasc_scales}\n")
        f.write(f"SASC Boundary Weight: {args.sasc_boundary_weight}\n")
        f.write(f"SASC MIND Weight: {args.sasc_mind_weight}\n")
        f.write(f"SASC MIND Scale: {args.sasc_mind_scale}\n")
        f.write(f"Use Residual Flow Pyramid: {args.use_residual_flow_pyramid}\n")
        f.write(f"Use Error-Guided Residual: {args.use_error_guided_residual}\n")
        f.write(f"Residual Flow Limit: {args.residual_flow_limit}\n")
        f.write(f"Residual Flow Reg Weight: {args.residual_flow_reg_weight}\n")
        f.write(f"Cross-Mamba Offset Limit: {args.cross_mamba_offset_limit}\n")
        f.write(f"Cross-Mamba Offset Smooth Kernel: {args.cross_mamba_offset_smooth_kernel}\n")
        f.write(f"Use SDMR: {args.use_sdmr}\n")
        f.write(f"SDMR Use MIND: {args.sdmr_use_mind}\n")
        f.write(f"SDMR Scale: {args.sdmr_scale}\n")
        f.write(f"SDMR Hidden Channels: {args.sdmr_hidden_channels}\n")
        f.write(f"SDMR Flow Limit: {args.sdmr_flow_limit}\n")
        f.write(f"SDMR Alpha: {args.sdmr_alpha}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Integration Steps: {args.integration_steps}\n")
        f.write(f"Arguments: {vars(args)}\n")

    # Initialize Logger
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_img_loss', 'train_grad_loss', 'train_sasc_loss', 'train_feature_edge_loss', 'val_dsc', 'val_hd95', 'val_jac', 'val_mag'])

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_dsc = 0.0
    loss_history = []
    val_dsc_history = []
    
    epoch_times = []
    
    # 记录训练前的初始 GPU 显存
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_num = epoch + 1
        boundary_loss_active = args.use_boundary_loss and (epoch_num > args.boundary_loss_start_epoch)
        active_boundary_loss_weight = args.boundary_loss_weight if boundary_loss_active else 0.0
        sasc_active = args.use_sasc and epoch_num >= args.sasc_start_epoch
        if sasc_active and args.sasc_ramp_epochs > 0:
            sasc_ramp = min(1.0, (epoch_num - args.sasc_start_epoch + 1) / args.sasc_ramp_epochs)
        else:
            sasc_ramp = 1.0 if sasc_active else 0.0
        active_sasc_loss_weight = args.sasc_weight * sasc_ramp
        feature_edge_active = args.use_feature_edge_loss and (epoch_num >= args.feature_edge_start_epoch)
        active_feature_edge_loss_weight = base_feature_edge_loss_weight if feature_edge_active else 0.0
        pyramid_image_active = epoch_num >= args.pyramid_image_start_epoch
        active_pyramid_image_weight = args.pyramid_image_weight if pyramid_image_active else 0.0
        
        avg_loss, train_img_loss, train_grad_loss, train_sasc_loss, train_feature_edge_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            pyramid_weight=args.pyramid_weight,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            use_mask=args.use_mask,
            loss_type=args.loss.lower(),
            image_loss_fn_coarse=image_loss_fn_coarse,
            boundary_loss_fn=boundary_loss_fn,
            boundary_loss_weight=active_boundary_loss_weight,
            boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
            boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
            sasc_loss_fn=sasc_loss_fn,
            sasc_loss_weight=active_sasc_loss_weight,
            feature_edge_loss_weight=active_feature_edge_loss_weight,
            feature_edge_indices=feature_edge_indices,
            pyramid_image_weight=active_pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=args.residual_flow_reg_weight,
        )
        loss_history.append(avg_loss)
        
        # Calculate Validation metrics (Fast: only DSC)
        _, _, val_dsc = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            compute_extra=False,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            boundary_loss_fn=boundary_loss_fn,
            boundary_loss_weight=active_boundary_loss_weight,
            boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
            boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
            sasc_loss_fn=sasc_loss_fn,
            sasc_loss_weight=active_sasc_loss_weight,
            feature_edge_loss_weight=active_feature_edge_loss_weight,
            feature_edge_indices=feature_edge_indices,
            pyramid_weight=args.pyramid_weight,
            pyramid_image_weight=active_pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=args.residual_flow_reg_weight,
            use_mask=args.use_mask,
            loss_type=args.loss.lower(),
            image_loss_fn_coarse=image_loss_fn_coarse,
        )
        
        # Decide if we need to compute heavy extra metrics (HD95, Jac, Mag)
        # Condition: 第1, 3, 5, 7个epoch，之后每10个epoch，或者dice达到最佳且大于0.77
        is_new_best = val_dsc > best_dsc
        condition_epoch = epoch_num in [1, 3, 5, 7] or epoch_num % 10 == 0
        condition_best = is_new_best and (val_dsc > 0.77)
        compute_extra = condition_epoch or condition_best
        
        if compute_extra:
            # Rerun validate to get the extra outputs. 
            # (Slight overhead of redoing model forward & DSC, but totally negligible compared to HD95 calculation)
            val_res_extra = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                compute_extra=True,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                boundary_loss_fn=boundary_loss_fn,
                boundary_loss_weight=active_boundary_loss_weight,
                boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
                boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
                sasc_loss_fn=sasc_loss_fn,
                sasc_loss_weight=active_sasc_loss_weight,
                feature_edge_loss_weight=active_feature_edge_loss_weight,
                feature_edge_indices=feature_edge_indices,
                pyramid_weight=args.pyramid_weight,
                residual_flow_reg_weight=args.residual_flow_reg_weight,
                use_mask=args.use_mask,
                loss_type=args.loss.lower(),
            )
            _, _, _, val_hd95, val_jac, val_mag = val_res_extra
        else:
            val_hd95, val_jac, val_mag = np.nan, np.nan, np.nan
            
        val_dsc_history.append(val_dsc)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        if compute_extra:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Img: {train_img_loss:.6f}, Grad: {train_grad_loss:.6f}, SASC: {train_sasc_loss:.6f}, FeatureEdge: {train_feature_edge_loss:.6f}, Val DSC: {val_dsc:.6f}, HD95: {val_hd95:.2f}, Jac: {val_jac:.4f}, Mag: {val_mag:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')
        else:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Img: {train_img_loss:.6f}, Grad: {train_grad_loss:.6f}, SASC: {train_sasc_loss:.6f}, FeatureEdge: {train_feature_edge_loss:.6f}, Val DSC: {val_dsc:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')

        # Save visualizations periodically or when a new best model is found to reduce epoch overhead
        save_vis = compute_extra
        if save_vis:
            try:
                save_qualitative_results(
                    model,
                    val_set,
                    out_path.parent,
                    epoch=epoch+1,
                    device=device,
                    boundary_kernel=args.boundary_kernel,
                    boundary_smooth_kernel=args.boundary_smooth_kernel,
                    boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
                    boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
                )
            except Exception as e:
                print(f"Failed to save visualization: {e}")

        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_loss:.6f}", 
                f"{train_img_loss:.6f}",
                f"{train_grad_loss:.6f}",
                f"{train_sasc_loss:.6f}",
                f"{train_feature_edge_loss:.6f}",
                f"{val_dsc:.6f}", 
                f"{val_hd95:.2f}" if compute_extra else "",
                f"{val_jac:.6f}" if compute_extra else "",
                f"{val_mag:.6f}" if compute_extra else ""
            ])

        # Early stopping check based on average loss
        if len(loss_history) >= args.warm_start + args.patience + 1:
            recent_losses = loss_history[-args.patience:]
            best_past_loss = min(loss_history[:-args.patience])
            if all(max(best_past_loss - loss, 0) < args.threshold for loss in recent_losses):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = out_path.parent / f'{out_path.stem}_epoch{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        if is_new_best:
            best_dsc = val_dsc
            best_path = out_path.parent / f'{out_path.stem}_best.pt'
            torch.save(model.state_dict(), best_path)
            print(f'Saved new best model with DSC: {best_dsc:.6f} (HD95: {val_hd95:.2f}, Jac: {val_jac:.4f})')

        if (epoch + 1) % 10 == 0:
            try:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:red'
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Train Loss', color=color)
                ax1.plot(range(1, epoch + 2), loss_history, color=color, marker='o', markersize=4, label='Train Loss')
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Val DSC', color=color)
                ax2.plot(range(1, epoch + 2), val_dsc_history, color=color, marker='s', markersize=4, label='Val DSC')
                ax2.tick_params(axis='y', labelcolor=color)
                
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left' if loss_history[0] > loss_history[-1] else 'center right')
                
                plt.title(f'Learning Curves (Epoch 1 to {epoch + 1})')
                fig.tight_layout()
                ax1.grid(True, linestyle='--', alpha=0.6)
                
                plot_path = out_path.parent / f'learning_curves_epoch{epoch + 1}.png'
                plt.savefig(str(plot_path), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to save learning curve plot: {e}")

    # Save final model
    torch.save(model.state_dict(), out_path)
    print(f'Final model saved to {out_path}')
    
    finish_utc = datetime.datetime.utcnow()
    finish_beijing = finish_utc + datetime.timedelta(hours=8)
    finish_timestamp = finish_beijing.strftime('%Y%m%d_%H%M%S')
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    final_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
    
    with open(config_file, 'a') as f:
        f.write(f"End Time: {finish_timestamp}\n")
        f.write(f"Average Epoch Time: {avg_epoch_time:.2f} s\n")
        f.write(f"Peak GPU Memory: {final_peak_mem:.2f} MB\n")

if __name__ == '__main__':
    main()
