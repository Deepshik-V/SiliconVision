# SiliconVision Model Checkpoints

This directory contains trained PyTorch model checkpoints for the **SiliconVision** semiconductor image restoration pipeline.

---

## 1. Trained Baseline Checkpoint

* **Filename**: `best_model.pth`
* **Architecture**: `BaselineSemiconNet` / `SiliconVisionRestorationNet`
* **Model Size / Parameters**: `18,211,009 parameters (18.21M)`
* **File Size**: `~219 MB`
* **Input Resolution**: `128 × 128` (`float32`, 1-channel grayscale)
* **Output Resolution**: `256 × 256` (`float32`, 1-channel grayscale, $2\times$ Super-Resolution)
* **Measured Validation Benchmark**:
  * **PSNR**: **23.11 dB**
  * **SSIM**: **0.9269**
  * **Inference Speed**: **~220 ms / image** (CPU)

---

## 2. Checkpoint Structure

The `.pth` checkpoint dictionary contains:
```python
{
    "epoch": 30,
    "model_state_dict": { ... },       # Model layer weights and biases
    "optimizer_state_dict": { ... },   # AdamW optimizer momentum states
    "val_psnr": 23.11,                 # Best validation PSNR achieved
    "val_ssim": 0.9269,                # Best validation SSIM achieved
    "norm_method": "per_image"         # Preprocessing dynamic range calibration
}
```

---

## 3. How the Code Loads the Checkpoint

The model is instantiated and loaded in Python via `models.model`:

```python
import torch
from models.model import BaselineSemiconNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)

checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()
```

---

## 4. Git Large File Exclusion Notice

In accordance with standard GitHub repository best practices, large binary weight files (`*.pth`, `*.pt` > 100 MB) are excluded from direct git commits via `.gitignore`. 

To obtain or reproduce weights:
* Keep your locally trained `best_model.pth` inside this `checkpoints/` folder.
* Alternatively, train a fresh model using:
  ```bash
  python train.py --epochs 30 --batch_size 8 --device cuda
  ```
