"""
SiliconVision Model Training Script
===================================
Reproducible PyTorch training pipeline for Semiconductor Image Restoration.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model import BaselineSemiconNet
from dataset import FastSemiconDataset, preload_raw_data
from normalization import get_scaler
from losses import CompositeRestorationLoss
from utils import calculate_psnr, calculate_ssim, set_seed
from config import Config

def train(args):
    print("=" * 75)
    print("SILICONVISION: SEMICONDUCTOR RESTORATION MODEL TRAINING")
    print(f"--> Device:         {args.device}")
    print(f"--> Epochs:         {args.epochs}")
    print(f"--> Batch Size:     {args.batch_size}")
    print(f"--> Learning Rate:  {args.lr}")
    print(f"--> Checkpoint Dir: {args.save_dir}")
    print("=" * 75)

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load dataset split manifest
    if not os.path.exists(args.split_manifest):
        raise FileNotFoundError(f"Split manifest not found at: {args.split_manifest}")

    with open(args.split_manifest, "r") as f:
        split_data = json.load(f)

    train_files = split_data["train_files"]
    val_files = split_data["val_files"]
    print(f"[+] Loaded Split: {len(train_files)} Train pairs | {len(val_files)} Validation pairs")

    scaler = get_scaler(args.norm_method)

    # Preload arrays into RAM
    print("--> Preloading validation and training samples into RAM...")
    val_cache = preload_raw_data(args.lr_dir, args.gt_dir, val_files)
    train_cache = preload_raw_data(args.lr_dir, args.gt_dir, train_files)

    train_ds = FastSemiconDataset(
        args.lr_dir, args.gt_dir, train_files, scaler,
        patch_size_lr=args.patch_size, scale_factor=2,
        is_train=True, use_augmentation=True, cached_data=train_cache
    )
    val_ds = FastSemiconDataset(
        args.lr_dir, args.gt_dir, val_files, scaler,
        patch_size_lr=0, scale_factor=2,
        is_train=False, use_augmentation=False, cached_data=val_cache
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Instantiate Model
    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[+] Instantiated BaselineSemiconNet ({param_count:,} parameters)")

    # Optimizer & Criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CompositeRestorationLoss(w_pixel=1.0, w_ssim=0.5, w_fft=0.1, w_sobel=0.2, in_channels=1).to(device)

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        epoch_start = time.time()

        for step, (lr_patch, gt_patch) in enumerate(train_loader):
            lr_patch = lr_patch.to(device)
            gt_patch = gt_patch.to(device)

            optimizer.zero_grad()
            pred = model(lr_patch)
            loss_dict = criterion(pred, gt_patch)
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        train_time = time.time() - epoch_start
        mean_train_loss = float(np.mean(train_losses))

        # Validation Pass
        model.eval()
        val_psnrs, val_ssims = [], []
        with torch.no_grad():
            for lr_full, gt_full in val_loader:
                lr_full = lr_full.to(device)
                gt_full = gt_full.to(device)

                pred_full = torch.clamp(model(lr_full), 0.0, 1.0)
                p = calculate_psnr(pred_full, gt_full)
                s = calculate_ssim(pred_full.squeeze().cpu().numpy(), gt_full.squeeze().cpu().numpy())
                val_psnrs.append(p)
                val_ssims.append(s)

        mean_psnr = float(np.mean(val_psnrs))
        mean_ssim = float(np.mean(val_ssims))

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Val PSNR: {mean_psnr:.2f} dB | "
            f"Val SSIM: {mean_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {train_time:.1f}s"
        )

        # Checkpoint Best Model
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_ssim = mean_ssim
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": best_psnr,
                "val_ssim": best_ssim,
                "norm_method": args.norm_method
            }, ckpt_path)
            print(f"  [+] Saved New Best Checkpoint: {ckpt_path} (PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f})")

    print("=" * 75)
    print(f"TRAINING COMPLETE: Best PSNR = {best_psnr:.2f} dB, Best SSIM = {best_ssim:.4f}")
    print("=" * 75)

def main():
    config = Config().resolve_paths()
    parser = argparse.ArgumentParser(description="SiliconVision Model Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--patch_size", type=int, default=64, help="LR patch crop size")
    parser.add_argument("--norm_method", type=str, default="per_image", help="Dynamic range scaler")
    parser.add_argument("--lr_dir", type=str, default=config.raw_train_lr_dir, help="Training NoisyLR directory")
    parser.add_argument("--gt_dir", type=str, default=config.raw_train_gt_dir, help="Training Ground Truth directory")
    parser.add_argument("--split_manifest", type=str, default=config.split_manifest, help="Path to split_indices.json")
    parser.add_argument("--save_dir", type=str, default=config.checkpoint_dir, help="Checkpoint output folder")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
