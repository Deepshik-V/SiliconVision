#!/usr/bin/env python3
"""
KLA SemiCon AI Hackathon - Official Submission Execution Script
=============================================================
Project: SiliconVision
Task: Semiconductor Image Restoration (NoisyLR 128x128 -> Restored 256x256)

Execution Syntax:
    python run.py <input-dir> <output-dir>

Example:
    python run.py ./input_dir ./output_dir
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# Ensure local models package is resolvable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.model import load_restoration_model
from models.normalization import PerImageRobustScaler

TARGET_INPUT_SHAPE = (128, 128)
TARGET_OUTPUT_SHAPE = (256, 256)

def run_pipeline(input_dir: str, output_dir: str):
    """
    Executes batch restoration on all .npy files in input_dir and saves outputs to output_dir.
    """
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Device Selection (Auto-detect CUDA, seamless CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> [SiliconVision] Execution Device: {str(device).upper()}")

    # 2. Checkpoint Loading (100% offline, local checkpoint)
    checkpoint_path = os.path.join(SCRIPT_DIR, "models", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        # Fallback to root checkpoints dir if run from root
        alt_ckpt = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
        if os.path.exists(alt_ckpt):
            checkpoint_path = alt_ckpt

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Model checkpoint not found at: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"--> [SiliconVision] Loading trained model from: {checkpoint_path}")
    t_load_start = time.time()
    try:
        model = load_restoration_model(checkpoint_path=checkpoint_path, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}", file=sys.stderr)
        sys.exit(1)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"--> [SiliconVision] Model initialized ({param_count:,} parameters) in {(time.time() - t_load_start)*1000.0:.1f}ms")

    # 3. Preprocessor Initialisation
    scaler = PerImageRobustScaler(p_min=0.1, p_max=99.9)

    # 4. Discover and sort input files for deterministic ordering
    all_entries = os.listdir(input_dir)
    npy_files = sorted([f for f in all_entries if f.lower().endswith(".npy")])

    if len(npy_files) == 0:
        print(f"[WARNING] No .npy files found in input directory: {input_dir}")
        return

    print(f"--> [SiliconVision] Found {len(npy_files)} .npy files to restore.")
    t_proc_start = time.time()
    success_count = 0

    # 5. Process each file
    with torch.inference_mode():
        for idx, filename in enumerate(npy_files, start=1):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)

            # Ingest input array
            try:
                raw_lr = np.load(in_path)
            except Exception as e:
                print(f"[ERROR] Failed to read {in_path}: {e}", file=sys.stderr)
                sys.exit(1)

            # Validate input dimensions
            if raw_lr.ndim != 2 or raw_lr.shape != TARGET_INPUT_SHAPE:
                print(f"[ERROR] Invalid input shape for {filename}: expected {TARGET_INPUT_SHAPE}, got {raw_lr.shape}", file=sys.stderr)
                sys.exit(1)

            # Check for non-finite values in input
            if not np.isfinite(raw_lr).all():
                print(f"[ERROR] Input file {filename} contains NaN or Inf values.", file=sys.stderr)
                sys.exit(1)

            # Normalization Preprocessing
            norm_lr = scaler(np.expand_dims(raw_lr, -1))
            in_tensor = torch.from_numpy(norm_lr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Model Forward Pass
            pred_tensor = model(in_tensor)

            # Clamping to valid [0.0, 1.0] signal bounds
            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)

            # Post-Processing to 2D float32 numpy array
            pred_np = pred_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Strict Output Validation
            if pred_np.shape != TARGET_OUTPUT_SHAPE:
                print(f"[ERROR] Output shape mismatch for {filename}: expected {TARGET_OUTPUT_SHAPE}, got {pred_np.shape}", file=sys.stderr)
                sys.exit(1)

            if pred_np.dtype != np.float32:
                print(f"[ERROR] Output dtype mismatch for {filename}: expected float32, got {pred_np.dtype}", file=sys.stderr)
                sys.exit(1)

            if not np.isfinite(pred_np).all():
                print(f"[ERROR] Restored output for {filename} contains NaN or Inf!", file=sys.stderr)
                sys.exit(1)

            if pred_np.min() < 0.0 or pred_np.max() > 1.0:
                print(f"[ERROR] Restored output for {filename} out of bounds: min={pred_np.min()}, max={pred_np.max()}", file=sys.stderr)
                sys.exit(1)

            # Save strictly as .npy
            np.save(out_path, pred_np)
            success_count += 1

            if idx % 50 == 0 or idx == len(npy_files):
                print(f"    [{idx}/{len(npy_files)}] Processed {filename} -> {out_path}")

    total_time = time.time() - t_proc_start
    avg_latency = (total_time / max(1, success_count)) * 1000.0
    print("=" * 75)
    print(f"--> [SiliconVision] Restoration Completed Successfully!")
    print(f"--> Processed:        {success_count}/{len(npy_files)} files")
    print(f"--> Total Time:       {total_time:.2f}s")
    print(f"--> Average Latency:  {avg_latency:.2f} ms / image")
    print(f"--> Output Location:  {output_dir}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(
        description="SiliconVision: Official KLA Semiconductor Image Restoration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
    python run.py ./input_dir ./output_dir
        """
    )
    parser.add_argument("input_dir", type=str, help="Path to input directory containing degraded .npy files")
    parser.add_argument("output_dir", type=str, help="Path to output directory to store restored .npy files")
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
