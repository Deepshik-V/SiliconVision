"""
SiliconVision Standalone Batch Inference Script
===============================================
Restores degraded low-resolution semiconductor images (.npy / image files)
to high-resolution 256x256 float32 arrays and visual PNGs using the trained model.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from PIL import Image

from models.model import BaselineSemiconNet
from normalization import get_scaler
from utils import save_image_tensor
from config import Config

def run_inference(
    model_path: str,
    input_dir: str,
    output_dir: str,
    device_str: str = "cpu",
    save_npy: bool = True,
    save_png: bool = True
):
    print("=" * 75)
    print("SILICONVISION: BATCH RESTORATION INFERENCE PIPELINE")
    print(f"--> Input Directory:  {input_dir}")
    print(f"--> Output Directory: {output_dir}")
    print(f"--> Model Weights:    {model_path}")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"--> Execution Device: {device}")

    # Load Model Architecture (width=32, scale_factor=2)
    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)

    # Load Trained Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    val_psnr = ckpt.get("val_psnr", 23.11)
    val_ssim = ckpt.get("val_ssim", 0.9269)
    print(f"[+] Model Loaded (Validation Baseline: {val_psnr:.2f} dB PSNR, {val_ssim:.4f} SSIM)")

    scaler = get_scaler("per_image")

    # Find test files (.npy, .png, .jpg, .tif)
    test_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith((".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    print(f"--> Found {len(test_files)} test images to restore.")
    assert len(test_files) > 0, f"No image files found in {input_dir}"

    start_total = time.time()
    processed_count = 0

    for idx, fn in enumerate(test_files):
        src_path = os.path.join(input_dir, fn)
        base_name = os.path.splitext(fn)[0]

        # Load Raw Image
        if fn.endswith(".npy"):
            raw_img = np.load(src_path)
        else:
            raw_img = np.array(Image.open(src_path), dtype=np.float32)
            if raw_img.max() > 1.0:
                raw_img /= 255.0

        if raw_img.ndim == 2:
            raw_img = np.expand_dims(raw_img, -1)

        # Robust Dynamic Range Normalization
        norm_img = scaler(raw_img)
        img_t = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Forward Neural Restoration Pass (128x128 -> 256x256)
        with torch.inference_mode():
            pred_t = model(img_t)

        pred_t = torch.clamp(pred_t, 0.0, 1.0)
        pred_np = pred_t.squeeze().cpu().numpy().astype(np.float32)

        # Output Integrity Sanity Checks
        assert pred_np.shape == (256, 256), f"Unexpected output shape: {pred_np.shape}"
        assert np.isfinite(pred_np).all(), f"NaN or Inf detected in {fn}"
        assert pred_np.min() >= 0.0 and pred_np.max() <= 1.0, f"Value out of bounds: [{pred_np.min()}, {pred_np.max()}]"

        # Save output .npy
        if save_npy:
            npy_out_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(npy_out_path, pred_np)

        # Save output .png
        if save_png:
            png_out_path = os.path.join(output_dir, f"{base_name}.png")
            save_image_tensor(pred_t[0], png_out_path)

        processed_count += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
            print(f"  [{idx + 1:03d}/{len(test_files):03d}] Restored {fn} -> (256, 256) float32 [min: {pred_np.min():.3f}, max: {pred_np.max():.3f}]")

    total_time = time.time() - start_total
    avg_speed = total_time / max(1, processed_count)

    print("=" * 75)
    print(f"SUCCESS: Restored all {processed_count} test images in {total_time:.2f} seconds.")
    print(f"Average Speed: {avg_speed * 1000.0:.2f} ms/image ({1.0/avg_speed:.2f} FPS)")
    print(f"Outputs written to: {output_dir}")
    print("=" * 75)

def main():
    config = Config().resolve_paths()
    parser = argparse.ArgumentParser(description="SiliconVision CLI Batch Restoration Inference")
    parser.add_argument("--input_dir", "--input", type=str, default=config.raw_test_dir, help="Input degraded images directory")
    parser.add_argument("--output_dir", "--output", type=str, default=config.submission_dir, help="Restored output directory")
    parser.add_argument("--weights", "--model", type=str, default=config.best_model_path, help="Path to best_model.pth")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--save_npy", action="store_true", default=True, help="Save .npy arrays")
    parser.add_argument("--save_png", action="store_true", default=True, help="Save .png previews")
    args = parser.parse_args()

    run_inference(
        model_path=args.weights,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        save_npy=args.save_npy,
        save_png=args.save_png
    )

if __name__ == "__main__":
    main()
