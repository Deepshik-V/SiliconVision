"""
SiliconVision Preprocessing & Dynamic Range Normalization
=========================================================
Robust quantile percentile scaling for semiconductor inspection images.
Handles out-of-bound dynamic range speckle noise without hard clipping loss.
"""

import numpy as np

class PerImageRobustScaler:
    """
    Robust Percentile Scaler:
    Calculates dynamic 0.1th and 99.9th percentiles independently per image.
    Preserves fine line edge gradients while mitigating multiplicative speckle spikes.
    """
    def __init__(self, p_min: float = 0.1, p_max: float = 99.9, eps: float = 1e-6):
        self.p_min = p_min
        self.p_max = p_max
        self.eps = eps

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        v_min = np.percentile(img_np, self.p_min)
        v_max = np.percentile(img_np, self.p_max)
        clipped = np.clip(img_np, v_min, v_max)
        norm = (clipped - v_min) / max(v_max - v_min, self.eps)
        return norm.astype(np.float32)

def get_scaler():
    return PerImageRobustScaler(p_min=0.1, p_max=99.9)
