import numpy as np

def calculate_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    """
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two 2D numpy arrays.
    """
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.mean((img1 - mu1) ** 2)
    sigma2_sq = np.mean((img2 - mu2) ** 2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(num / den)

def evaluate_restoration(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Computes comprehensive restoration metrics when GT is provided.
    """
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    
    psnr_val = calculate_psnr(pred, gt)
    ssim_val = calculate_ssim(pred, gt)
    mae_val = float(np.mean(np.abs(pred - gt)))
    mse_val = float(np.mean((pred - gt) ** 2))

    # Delta analysis
    return {
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "mae": round(mae_val, 5),
        "mse": round(mse_val, 6),
        "psnr_quality": "Excellent" if psnr_val >= 25.0 else ("Good" if psnr_val >= 22.0 else "Fair"),
        "ssim_quality": "High Fidelity" if ssim_val >= 0.90 else "Moderate Fidelity"
    }
