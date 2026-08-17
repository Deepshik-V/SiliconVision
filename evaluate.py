"""
SiliconVision Quantitative Metrology Evaluation Script
======================================================
Computes Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
Mean Absolute Error (MAE), and Mean Squared Error (MSE) when Ground Truth is available.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

from models.model import BaselineSemiconNet
from dataset import FastSemiconDataset, preload_raw_data
from normalization import get_scaler
from utils import calculate_psnr, calculate_ssim
from config import Config

def evaluate_model_on_split(model_path: str, lr_dir: str, gt_dir: str, split_manifest: str, device_str: str = "cpu"):
    print("=" * 75)
    print("SILICONVISION: QUANTITATIVE METROLOGY VALIDATION BENCHMARK")
    print(f"--> Model Checkpoint: {model_path}")
    print(f"--> Split Manifest:   {split_manifest}")
    print(f"--> Device:           {device_str}")
    print("=" * 75)

    if not os.path.exists(split_manifest):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

    with open(split_manifest, "r") as f:
        split_data = json.load(f)

    val_files = split_data["val_files"]
    print(f"[+] Loaded {len(val_files)} held-out validation sample paths.")

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    scaler = get_scaler("per_image")
    val_cache = preload_raw_data(lr_dir, gt_dir, val_files)
    val_ds = FastSemiconDataset(
        lr_dir, gt_dir, val_files, scaler,
        patch_size_lr=0, scale_factor=2,
        is_train=False, use_augmentation=False, cached_data=val_cache
    )

    psnrs, ssims, maes, mses, latencies = [], [], [], [], []

    print("--> Running validation evaluation across all samples...")
    with torch.inference_mode():
        for idx in range(len(val_ds)):
            lr, gt = val_ds[idx]
            lr_t = lr.unsqueeze(0).to(device)
            gt_t = gt.unsqueeze(0).to(device)

            t0 = time.time()
            pred_t = model(lr_t)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000.0)

            pred_clamped = torch.clamp(pred_t, 0.0, 1.0)
            pred_np = pred_clamped.squeeze().cpu().numpy()
            gt_np = gt_t.squeeze().cpu().numpy()

            p = calculate_psnr(pred_clamped, gt_t)
            s = calculate_ssim(pred_np, gt_np)
            mae = float(np.mean(np.abs(pred_np - gt_np)))
            mse = float(np.mean((pred_np - gt_np) ** 2))

            psnrs.append(p)
            ssims.append(s)
            maes.append(mae)
            mses.append(mse)

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_mae = float(np.mean(maes))
    mean_mse = float(np.mean(mses))
    avg_latency = float(np.mean(latencies))

    print("\n" + "=" * 75)
    print("BENCHMARK EVALUATION SCORECARD:")
    print("=" * 75)
    print(f"Validation Samples Evaluated: {len(val_ds)} images")
    print(f"Mean PSNR:                    {mean_psnr:.4f} dB")
    print(f"Mean SSIM:                    {mean_ssim:.4f}")
    print(f"Mean Absolute Error (MAE):    {mean_mae:.5f}")
    print(f"Mean Squared Error (MSE):     {mean_mse:.6f}")
    print(f"Average Inference Latency:    {avg_latency:.2f} ms / image ({1000.0/avg_latency:.2f} FPS)")
    print("=" * 75)

    return {
        "val_samples": len(val_ds),
        "mean_psnr": round(mean_psnr, 2),
        "mean_ssim": round(mean_ssim, 4),
        "mean_mae": round(mean_mae, 5),
        "mean_mse": round(mean_mse, 6),
        "avg_latency_ms": round(avg_latency, 2)
    }

def evaluate_predicted_folder(pred_dir: str, gt_dir: str):
    print("=" * 75)
    print("EVALUATING PRE-GENERATED PREDICTIONS AGAINST GROUND TRUTH")
    print(f"--> Predictions Dir:  {pred_dir}")
    print(f"--> Ground Truth Dir: {gt_dir}")
    print("=" * 75)

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".npy")])
    psnrs, ssims, maes, mses = [], [], [], []

    for fn in pred_files:
        pred_p = os.path.join(pred_dir, fn)
        gt_p = os.path.join(gt_dir, fn)
        if not os.path.exists(gt_p):
            continue

        pred_arr = np.load(pred_p).squeeze()
        gt_arr = np.load(gt_p).squeeze()

        p = calculate_psnr(torch.from_numpy(pred_arr).unsqueeze(0).unsqueeze(0), torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0))
        s = calculate_ssim(pred_arr, gt_arr)
        mae = float(np.mean(np.abs(pred_arr - gt_arr)))
        mse = float(np.mean((pred_arr - gt_arr) ** 2))

        psnrs.append(p)
        ssims.append(s)
        maes.append(mae)
        mses.append(mse)

    print(f"[+] Evaluated {len(psnrs)} paired images.")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} dB | Mean SSIM: {np.mean(ssims):.4f}")

def main():
    config = Config().resolve_paths()
    parser = argparse.ArgumentParser(description="SiliconVision Quantitative Evaluation Benchmark")
    parser.add_argument("--weights", type=str, default=config.best_model_path, help="Path to best_model.pth")
    parser.add_argument("--lr_dir", type=str, default=config.raw_train_lr_dir, help="NoisyLR directory")
    parser.add_argument("--gt_dir", type=str, default=config.raw_train_gt_dir, help="Ground Truth directory")
    parser.add_argument("--split_manifest", type=str, default=config.split_manifest, help="Path to split_indices.json")
    parser.add_argument("--pred_dir", type=str, default=None, help="Optional pre-generated predictions folder")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    if args.pred_dir:
        evaluate_predicted_folder(args.pred_dir, args.gt_dir)
    else:
        evaluate_model_on_split(args.weights, args.lr_dir, args.gt_dir, args.split_manifest, args.device)

if __name__ == "__main__":
    main()
