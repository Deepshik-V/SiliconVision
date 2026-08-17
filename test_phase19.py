import os
import sys
import json
import torch
import numpy as np

def run_lightweight_test():
    print("=" * 80)
    print("PHASE 19: FINAL LIGHTWEIGHT TEST & VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Imports
    print("--> 1. Testing Module Imports...")
    from models.model import BaselineSemiconNet, SiliconVisionRestorationNet, create_model
    from dataset import FastSemiconDataset
    from normalization import get_scaler
    from losses import CompositeRestorationLoss
    from utils import calculate_psnr, calculate_ssim
    from config import Config
    print("    [+] All modules imported successfully.")

    # 2. Checkpoint Loading
    print("--> 2. Testing Checkpoint Loading...")
    config = Config().resolve_paths()
    device = torch.device("cpu")
    model = create_model(config.best_model_path, device="cpu")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    [+] Checkpoint loaded: {config.best_model_path} ({param_count:,} params)")
    assert param_count == 18211009, f"Parameter count mismatch: {param_count}"

    # 3. 1-Sample Inference
    print("--> 3. Testing 1-Sample Inference...")
    demo_lr_path = os.path.join(config.project_root, "backend", "demo_samples", "sample_01_lr.npy")
    scaler = get_scaler("per_image")
    
    if os.path.exists(demo_lr_path):
        raw_lr = np.load(demo_lr_path)
    else:
        raw_lr = np.random.randn(128, 128).astype(np.float32)

    norm_lr = scaler(np.expand_dims(raw_lr, -1))
    in_tensor = torch.from_numpy(norm_lr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        pred_tensor = model(in_tensor)
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    
    pred_np = pred_tensor.squeeze().cpu().numpy().astype(np.float32)
    print(f"    [+] Input Shape:  {raw_lr.shape} (dtype: {raw_lr.dtype})")
    print(f"    [+] Output Shape: {pred_np.shape} (dtype: {pred_np.dtype})")
    print(f"    [+] Value Range:  [{pred_np.min():.6f}, {pred_np.max():.6f}]")
    print(f"    [+] NaN Count:    {np.isnan(pred_np).sum()}")
    print(f"    [+] Inf Count:    {np.isinf(pred_np).sum()}")

    assert pred_np.shape == (256, 256), f"Shape mismatch: {pred_np.shape}"
    assert pred_np.dtype == np.float32, f"Dtype mismatch: {pred_np.dtype}"
    assert not np.isnan(pred_np).any(), "NaN detected!"
    assert not np.isinf(pred_np).any(), "Inf detected!"
    assert pred_np.min() >= 0.0 and pred_np.max() <= 1.0, "Range out of bounds!"

    # 4. Backend Service Test
    print("--> 4. Testing Backend Service Singleton...")
    from backend.inference_service import RestorationService
    service = RestorationService.get_instance()
    info = service.get_info()
    print("    [+] Model Name:", info["model_name"])
    print("    [+] Status:    ", info["status"])
    print("    [+] Device:    ", info["device"])

    print("=" * 80)
    print("ALL LIGHTWEIGHT VERIFICATION CHECKS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_lightweight_test()
