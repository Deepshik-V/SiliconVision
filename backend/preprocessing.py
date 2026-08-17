import io
import base64
import numpy as np
import torch
from PIL import Image

def load_image_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Parses uploaded file bytes into a 2D float32 numpy array.
    Supports .npy and standard image formats (.png, .jpg, .tif).
    Preserves exact raw numerical data without modification.
    """
    fn_lower = filename.lower()
    if fn_lower.endswith(".npy"):
        arr = np.load(io.BytesIO(file_bytes))
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        arr = np.array(pil_img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    elif arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr.astype(np.float32)

def extract_image_metadata(arr: np.ndarray, filename: str) -> dict:
    """
    Extracts statistical and telemetry metadata for an input array.
    """
    h, w = arr.shape
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    v_mean = float(np.mean(arr))
    v_std = float(np.std(arr))
    
    below_zero_pct = float(np.mean(arr < 0.0) * 100.0)
    above_one_pct = float(np.mean(arr > 1.0) * 100.0)
    has_overflow = below_zero_pct > 0.0 or above_one_pct > 0.0

    return {
        "filename": filename,
        "height": h,
        "width": w,
        "shape": f"{h} × {w}",
        "dtype": str(arr.dtype),
        "min_value": round(v_min, 4),
        "max_value": round(v_max, 4),
        "mean_value": round(v_mean, 4),
        "std_value": round(v_std, 4),
        "below_zero_pct": round(below_zero_pct, 2),
        "above_one_pct": round(above_one_pct, 2),
        "has_overflow": has_overflow
    }

def array_to_base64_png(arr: np.ndarray, is_gt_or_restored: bool = False, upscale_preview: bool = True) -> str:
    """
    Converts a 2D float32 numpy array into a base64 encoded high-resolution PNG preview.
    Applies high-fidelity contrast & dynamic range normalization ONLY for visual rendering,
    leaving the underlying raw numerical array untouched.
    """
    if is_gt_or_restored:
        # Array is already in [0, 1] range; clip safely and map to [0, 255]
        vis_arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        # Input NoisyLR contains speckle noise bursts outside [0, 1]
        # Use robust percentile contrast stretching (0.5th to 99.5th percentile) for crisp visualization
        p_low = np.percentile(arr, 0.5)
        p_high = np.percentile(arr, 99.5)
        
        if p_high > p_low:
            stretched = (np.clip(arr, p_low, p_high) - p_low) / (p_high - p_low)
            vis_arr = (stretched * 255.0).astype(np.uint8)
        else:
            vis_arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(vis_arr, mode="L")

    # If input is 128x128, produce a high-quality bicubic preview upscaled to 256x256
    # so both 128x128 input and 256x256 output render at identical smooth visual scale!
    if upscale_preview and pil_img.size == (128, 128):
        pil_img = pil_img.resize((256, 256), resample=Image.Resampling.BICUBIC)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
