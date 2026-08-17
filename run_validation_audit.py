import os
import json
import time
import numpy as np
import torch

from models.model import BaselineSemiconNet
from dataset import FastSemiconDataset, preload_raw_data
from normalization import get_scaler
from utils import calculate_psnr, calculate_ssim
from config import Config

def run_audit():
    config = Config().resolve_paths()
    with open(config.split_manifest, "r") as f:
        split = json.load(f)

    val_files = split["val_files"]
    scaler = get_scaler("per_image")

    val_cache = preload_raw_data(config.raw_train_lr_dir, config.raw_train_gt_dir, val_files)
    val_ds = FastSemiconDataset(
        config.raw_train_lr_dir, config.raw_train_gt_dir, val_files, scaler,
        patch_size_lr=0, scale_factor=2, is_train=False, use_augmentation=False, cached_data=val_cache
    )

    device = torch.device("cpu")
    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)
    ckpt_path = config.best_model_path

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    psnrs, ssims, maes, mses, times = [], [], [], [], []

    with torch.inference_mode():
        for i in range(len(val_ds)):
            lr, gt = val_ds[i]
            lr_t = lr.unsqueeze(0).to(device)
            gt_t = gt.unsqueeze(0).to(device)

            t0 = time.time()
            pred = model(lr_t)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)

            pred_clamped = torch.clamp(pred, 0.0, 1.0)
            p = calculate_psnr(pred_clamped, gt_t)
            s = calculate_ssim(pred_clamped.squeeze().cpu().numpy(), gt_t.squeeze().cpu().numpy())
            mae = float(torch.mean(torch.abs(pred_clamped - gt_t)).item())
            mse = float(torch.mean((pred_clamped - gt_t) ** 2).item())

            psnrs.append(p)
            ssims.append(s)
            maes.append(mae)
            mses.append(mse)

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_mae = float(np.mean(maes))
    mean_mse = float(np.mean(mses))
    avg_time = float(np.mean(times))

    print("=" * 80)
    print("GROUND TRUTH VALIDATION METRICS (200 HELD-OUT VALIDATION SAMPLES):")
    print("=" * 80)
    print(f"Evaluated Images:          {len(val_ds)}")
    print(f"Mean Validation PSNR:      {mean_psnr:.4f} dB  (Rounded: {mean_psnr:.2f} dB)")
    print(f"Mean Validation SSIM:      {mean_ssim:.4f}")
    print(f"Mean Absolute Error (MAE): {mean_mae:.5f}")
    print(f"Mean Squared Error (MSE):  {mean_mse:.6f}")
    print(f"Average Inference Latency: {avg_time:.2f} ms / image")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
