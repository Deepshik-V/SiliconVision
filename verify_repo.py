import sys
import os
import json
import torch

def verify_all():
    print("=" * 75)
    print("SILICONVISION: END-TO-END REPOSITORY VERIFICATION AUDIT")
    print("=" * 75)

    # 1. Test Model Imports
    print("--> 1. Testing Model Imports...")
    from models.model import BaselineSemiconNet, SiliconVisionRestorationNet, create_model
    from dataset import FastSemiconDataset, SemiconDataset
    from losses import CompositeRestorationLoss
    from normalization import get_scaler
    from utils import calculate_psnr, calculate_ssim
    print("    [+] All module imports successful!")

    # 2. Test Checkpoint Loading
    print("--> 2. Testing Checkpoint Loading...")
    ckpt_path = "checkpoints/best_model.pth"
    device = torch.device("cpu")
    model = create_model(ckpt_path, device="cpu")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    [+] Loaded best_model.pth ({param_count:,} parameters).")
    assert param_count == 18211009, f"Parameter mismatch: {param_count}"

    # 3. Test 1-Sample Inference Forward Pass
    print("--> 3. Testing 1-Sample Inference...")
    dummy_input = torch.randn(1, 1, 128, 128)
    with torch.inference_mode():
        out = model(dummy_input)
    print(f"    [+] Input: {dummy_input.shape} -> Output: {out.shape}")
    assert out.shape == (1, 1, 256, 256), f"Output shape mismatch: {out.shape}"

    # 4. Test Backend Service
    print("--> 4. Testing Backend Inference Service...")
    from backend.inference_service import RestorationService
    service = RestorationService.get_instance()
    info = service.get_info()
    print("    [+] Backend Model Info:", info["model_name"], "| Device:", info["device"])

    print("=" * 75)
    print("ALL VERIFICATION CHECKS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    verify_all()
