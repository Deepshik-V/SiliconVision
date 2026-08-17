import numpy as np
import json
import os

class BaseScaler:
    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class RawScaler(BaseScaler):
    """
    Experiment 1A: Raw NoisyLR fed directly into the model.
    No scaling applied. The model's initial convolution learns the representation.
    """
    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        return img_np.astype(np.float32)

class GlobalMinMaxScaler(BaseScaler):
    """
    Experiment 1B: Global Dataset-Wide Calibration Scaling (Fixed Min-Max).
    Monotonically maps the entire training dynamic range [-0.278563, 2.158005] to [0.0, 1.0].
    Preserves all relative contrast differences across different semiconductor patterns without distortion.
    """
    def __init__(self, v_min: float = -0.278563, v_max: float = 2.158005):
        self.v_min = v_min
        self.v_max = v_max
        self.v_range = v_max - v_min

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        norm = (img_np - self.v_min) / self.v_range
        return norm.astype(np.float32)

class RobustPercentileScaler(BaseScaler):
    """
    Experiment 1C: Dataset-Wide Robust Percentile Scaling (Fixed P0.1 - P99.9).
    Maps the 0.1th percentile (-0.005412) to 99.9th percentile (1.328314) to [0.0, 1.0],
    clipping the extreme 0.2% outliers from speckle burst noise.
    """
    def __init__(self, p_min: float = -0.005412, p_max: float = 1.328314):
        self.p_min = p_min
        self.p_max = p_max
        self.p_range = p_max - p_min

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        clipped = np.clip(img_np, self.p_min, self.p_max)
        norm = (clipped - self.p_min) / self.p_range
        return norm.astype(np.float32)

class ZScoreScaler(BaseScaler):
    """
    Experiment 1D: Dataset-Wide Standardized / Z-Score Normalization.
    Normalizes input to zero mean and unit variance using dataset statistics:
    mean = 0.431935, std = 0.205888.
    """
    def __init__(self, mean: float = 0.431935, std: float = 0.205888):
        self.mean = mean
        self.std = std

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        norm = (img_np - self.mean) / self.std
        return norm.astype(np.float32)

class PerImageRobustScaler(BaseScaler):
    """
    Control (Current Baseline Scaler):
    Calculates dynamic 0.1th and 99.9th percentiles independently for EACH image.
    Used to reproduce and benchmark against the existing baseline implementation.
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

def get_scaler(method: str, stats_file: str = None) -> BaseScaler:
    """
    Factory function returning the configured normalization scaler.
    """
    method = method.lower()
    if method == "raw":
        return RawScaler()
    elif method in ["global_minmax", "minmax"]:
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
            return GlobalMinMaxScaler(v_min=stats["global_min"], v_max=stats["global_max"])
        return GlobalMinMaxScaler()
    elif method in ["robust_percentile", "percentile"]:
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
            return RobustPercentileScaler(p_min=stats["p01"], p_max=stats["p999"])
        return RobustPercentileScaler()
    elif method in ["zscore", "standardize"]:
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
            return ZScoreScaler(mean=stats["mean"], std=stats["std"])
        return ZScoreScaler()
    elif method in ["per_image", "baseline"]:
        return PerImageRobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
