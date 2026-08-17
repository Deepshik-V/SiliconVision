import os
import sys
import time
import json
import torch
import numpy as np

from config import Config
from normalization import get_scaler
from dataset import SemiconDataset
from models.baseline import BaselineSemiconNet
from losses import CompositeRestorationLoss
from utils import calculate_psnr, calculate_ssim, set_seed, save_image_tensor

def run_smoke_test():
    print("=" * 75)
    print("KLA SEMICON RESTORATION: 1-BATCH COMPREHENSIVE SMOKE TEST")
    print("=" * 75)

    start_time = time.time()
    config = Config()
    set_seed(config.seed)

    # 1. Check Split Manifest & Dataset Existence
    print("\n[Step 1/7] Verifying Dataset Split & Paths...")
    assert os.path.exists(config.split_manifest), f"Missing split manifest: {config.split_manifest}"
    with open(config.split_manifest, "r") as f:
        split_data = json.load(f)

    train_files = split_data["train_files"]
    val_files = split_data["val_files"]
    assert len(train_files) == 3000, f"Expected 3000 train files, got {len(train_files)}"
    assert len(val_files) == 200, f"Expected 200 val files, got {len(val_files)}"
    assert len(set(train_files).intersection(set(val_files))) == 0, "Data leakage detected: train and val overlap!"
    print(f"  --> Split Verified: {len(train_files)} Train, {len(val_files)} Val, 0 Overlap.")

    # 2. Test Normalization Scaler
    print("\n[Step 2/7] Testing Normalization Scaler...")
    scaler = get_scaler("global_minmax", stats_file=config.train_stats_file)
    test_arr = np.array([-0.278563, 0.431935, 2.158005], dtype=np.float32)
    norm_res = scaler(test_arr)
    assert abs(norm_res[0] - 0.0) < 1e-4, f"Min scaling error: {norm_res[0]}"
    assert abs(norm_res[2] - 1.0) < 1e-4, f"Max scaling error: {norm_res[2]}"
    print(f"  --> Global Min-Max scaling verified: [-0.2785, 2.1580] -> [{norm_res[0]:.4f}, {norm_res[2]:.4f}]")

    # 3. Create 1-Batch DataLoader
    print("\n[Step 3/7] Loading 1 Batch of Real Official Data...")
    smoke_batch_size = 4
    train_dataset = SemiconDataset(
        lr_dir=config.raw_train_lr_dir,
        gt_dir=config.raw_train_gt_dir,
        file_list=train_files[:smoke_batch_size],
        scaler=scaler,
        patch_size_lr=64,
        scale_factor=2,
        is_train=True,
        use_augmentation=True
    )
    val_dataset = SemiconDataset(
        lr_dir=config.raw_train_lr_dir,
        gt_dir=config.raw_train_gt_dir,
        file_list=val_files[:2],
        scaler=scaler,
        patch_size_lr=0,
        scale_factor=2,
        is_train=False,
        use_augmentation=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=smoke_batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    lr_batch, gt_batch = next(iter(train_loader))
    print(f"  --> Train Batch LR Shape: {lr_batch.shape}, GT Shape: {gt_batch.shape}")
    assert lr_batch.shape == (smoke_batch_size, 1, 64, 64), f"Incorrect LR shape: {lr_batch.shape}"
    assert gt_batch.shape == (smoke_batch_size, 1, 128, 128), f"Incorrect GT shape: {gt_batch.shape}"

    # 4. Model Architecture & Forward Pass
    print("\n[Step 4/7] Initializing Baseline Model & Testing Forward Pass...")
    device = torch.device("cpu")
    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  --> Baseline Model Initialized: {param_count:,} parameters ({param_count/1e6:.2f}M).")

    pred_batch = model(lr_batch)
    print(f"  --> Forward Pass Output Shape: {pred_batch.shape}")
    assert pred_batch.shape == gt_batch.shape, f"Shape mismatch: {pred_batch.shape} vs {gt_batch.shape}"

    # 5. Loss Function & Backward Pass
    print("\n[Step 5/7] Testing Composite Loss & Backward Gradient Propagation...")
    criterion = CompositeRestorationLoss(w_pixel=1.0, w_ssim=0.5, w_fft=0.1, w_sobel=0.2)
    loss, loss_dict = criterion(pred_batch, gt_batch)
    print(f"  --> Computed Loss: Total={loss.item():.4f}, Pixel={loss_dict['pixel']:.4f}, SSIM={loss_dict['ssim']:.4f}, FFT={loss_dict['fft']:.4f}, Sobel={loss_dict['sobel']:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    assert all(np.isfinite(g) for g in grad_norms), "NaN/Inf detected in gradients!"
    optimizer.step()
    print(f"  --> Backward Pass & Optimizer Step Successful. (Average Grad Norm: {np.mean(grad_norms):.4f})")

    # 6. Validation Pass & Metrics
    print("\n[Step 6/7] Testing Full-Resolution Validation Inference & Metric Evaluation...")
    model.eval()
    val_lr, val_gt = next(iter(val_loader))
    print(f"  --> Validation Input Full Shape: {val_lr.shape}")
    assert val_lr.shape == (1, 1, 128, 128), f"Expected (1, 1, 128, 128), got {val_lr.shape}"

    with torch.no_grad():
        val_pred = model(val_lr)
        val_pred_clamped = torch.clamp(val_pred, 0.0, 1.0)

    assert val_pred_clamped.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {val_pred_clamped.shape}"
    psnr_val = calculate_psnr(val_pred_clamped, val_gt)
    ssim_val = calculate_ssim(val_pred_clamped.squeeze().numpy(), val_gt.squeeze().numpy())
    print(f"  --> Full-Res Restoration Shape: {val_pred_clamped.shape} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    # 7. Checkpoint Saving & Verification
    print("\n[Step 7/7] Testing Checkpoint Serialization & Deserialization...")
    os.makedirs(config.output_dir, exist_ok=True)
    test_ckpt_path = os.path.join(config.output_dir, "smoke_test_ckpt.pth")
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "psnr": psnr_val,
        "ssim": ssim_val,
        "norm_method": "global_minmax"
    }, test_ckpt_path)
    assert os.path.exists(test_ckpt_path), "Failed to save test checkpoint!"

    # Reload checkpoint
    loaded = torch.load(test_ckpt_path, map_location="cpu")
    model.load_state_dict(loaded["model_state_dict"])
    print(f"  --> Checkpoint successfully written and reloaded from {test_ckpt_path}")

    # Clean up smoke test artifact
    if os.path.exists(test_ckpt_path):
        os.remove(test_ckpt_path)

    duration = time.time() - start_time
    print("=" * 75)
    print(f"SMOKE TEST PASSED ALL ASSERTIONS SUCCESSFULLY in {duration:.2f} seconds.")
    print("=" * 75)

if __name__ == "__main__":
    run_smoke_test()
