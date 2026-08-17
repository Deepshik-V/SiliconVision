import os
import random
import numpy as np
import torch
from PIL import Image

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()

def calculate_ssim(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """
    Standard SSIM metric between two 2D numpy arrays [0, 1].
    """
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = img1_np.mean()
    mu2 = img2_np.mean()

    sigma1_sq = ((img1_np - mu1) ** 2).mean()
    sigma2_sq = ((img2_np - mu2) ** 2).mean()
    sigma12 = ((img1_np - mu1) * (img2_np - mu2)).mean()

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(num / den)

def save_image_tensor(tensor: torch.Tensor, path: str):
    """
    Saves a (1, H, W) or (3, H, W) torch tensor [0.0, 1.0] to a PNG file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_np = tensor.detach().cpu().squeeze().numpy()
    img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(path)
