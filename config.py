import os
from dataclasses import dataclass

@dataclass
class Config:
    # Project Root Directory (relative to this file)
    project_root: str = os.path.abspath(os.path.dirname(__file__))

    # Data Directories (Configurable via Environment Variables or CLI)
    raw_train_lr_dir: str = os.environ.get(
        "KLA_TRAIN_LR_DIR",
        os.path.join(project_root, "data", "train", "NoisyLR")
    )
    raw_train_gt_dir: str = os.environ.get(
        "KLA_TRAIN_GT_DIR",
        os.path.join(project_root, "data", "train", "GT")
    )
    raw_test_dir: str = os.environ.get(
        "KLA_TEST_DIR",
        os.path.join(project_root, "data", "test", "NoisyLR")
    )

    # Split Manifest & Statistics
    split_manifest: str = os.path.join(project_root, "data", "splits", "split_indices.json")
    train_stats_path: str = os.path.join(project_root, "data", "splits", "train_stats.json")

    # Checkpoint and Output Directories
    checkpoint_dir: str = os.path.join(project_root, "checkpoints")
    best_model_path: str = os.path.join(project_root, "checkpoints", "best_model.pth")
    results_dir: str = os.path.join(project_root, "results")
    submission_dir: str = os.path.join(project_root, "results", "submission")

    # Model Parameters
    in_channels: int = 1
    out_channels: int = 1
    width: int = 32
    scale_factor: int = 2
    enc_blk_nums: tuple = (2, 2, 4, 6)
    middle_blk_num: int = 6
    dec_blk_nums: tuple = (2, 2, 2, 2)

    # Training Hyperparameters
    seed: int = 42
    batch_size: int = 8
    patch_size_lr: int = 64
    patch_size_gt: int = 128
    num_epochs: int = 30
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4

    # Loss Weights: Charbonnier + SSIM + FFT + Sobel
    w_charb: float = 1.0
    w_ssim: float = 0.5
    w_fft: float = 0.1
    w_sobel: float = 0.2

    # Normalization Strategy
    norm_method: str = "per_image"  # 'per_image', 'robust_percentile', 'zscore', 'global_minmax', 'raw'

    # Portable path resolution checking environment and common user download locations
    def resolve_paths(self):
        user_home = os.path.expanduser("~")
        if not os.path.exists(self.raw_train_lr_dir):
            fallback_lr = os.path.join(user_home, "Downloads", "train", "train", "NoisyLR")
            if os.path.exists(fallback_lr):
                self.raw_train_lr_dir = fallback_lr
        if not os.path.exists(self.raw_train_gt_dir):
            fallback_gt = os.path.join(user_home, "Downloads", "train", "train", "GT")
            if os.path.exists(fallback_gt):
                self.raw_train_gt_dir = fallback_gt
        if not os.path.exists(self.raw_test_dir):
            fallback_test = os.path.join(user_home, "Downloads", "Test_NoisyLR", "NoisyLR")
            if os.path.exists(fallback_test):
                self.raw_test_dir = fallback_test
        return self
