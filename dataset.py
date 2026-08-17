import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def preload_raw_data(lr_dir: str, gt_dir: str, file_list: list) -> dict:
    """
    Preloads numpy arrays into RAM to eliminate disk I/O bottlenecks during training.
    """
    cache = {}
    for fn in file_list:
        lr_p = os.path.join(lr_dir, fn)
        gt_p = os.path.join(gt_dir, fn)
        if os.path.exists(lr_p) and os.path.exists(gt_p):
            cache[fn] = (np.load(lr_p), np.load(gt_p))
    return cache

class FastSemiconDataset(Dataset):
    """
    High-Performance Semiconductor Restoration Dataset with in-memory caching,
    paired random spatial crop, and joint geometric augmentations.
    """
    def __init__(
        self,
        lr_dir: str,
        gt_dir: str,
        file_list: list,
        scaler,
        patch_size_lr: int = 64,
        scale_factor: int = 2,
        is_train: bool = True,
        use_augmentation: bool = True,
        cached_data: dict = None
    ):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.file_list = [f for f in file_list if cached_data is None or f in cached_data or os.path.exists(os.path.join(lr_dir, f))]
        self.scaler = scaler
        self.patch_size_lr = patch_size_lr
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.file_list)

    def _augment(self, lr: np.ndarray, gt: np.ndarray):
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.fliplr(lr)
            gt = np.fliplr(gt)
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flipud(lr)
            gt = np.flipud(gt)
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k)
            gt = np.rot90(gt, k)
        return lr.copy(), gt.copy()

    def __getitem__(self, idx: int):
        fn = self.file_list[idx]

        if self.cached_data is not None and fn in self.cached_data:
            raw_lr, raw_gt = self.cached_data[fn]
        else:
            raw_lr = np.load(os.path.join(self.lr_dir, fn))
            raw_gt = np.load(os.path.join(self.gt_dir, fn))

        if raw_lr.ndim == 2:
            raw_lr = np.expand_dims(raw_lr, -1)
        if raw_gt.ndim == 2:
            raw_gt = np.expand_dims(raw_gt, -1)

        # Apply robust dynamic range scaler to input
        norm_lr = self.scaler(raw_lr)
        norm_gt = raw_gt.astype(np.float32)

        # Training patch crop
        if self.is_train and self.patch_size_lr > 0:
            H_lr, W_lr, _ = norm_lr.shape
            ps_lr = self.patch_size_lr
            ps_gt = ps_lr * self.scale_factor

            top_lr = random.randint(0, max(0, H_lr - ps_lr))
            left_lr = random.randint(0, max(0, W_lr - ps_lr))

            top_gt = top_lr * self.scale_factor
            left_gt = left_lr * self.scale_factor

            patch_lr = norm_lr[top_lr:top_lr + ps_lr, left_lr:left_lr + ps_lr, :]
            patch_gt = norm_gt[top_gt:top_gt + ps_gt, left_gt:left_gt + ps_gt, :]

            if self.use_augmentation:
                patch_lr, patch_gt = self._augment(patch_lr, patch_gt)

            t_lr = torch.from_numpy(patch_lr).permute(2, 0, 1).float()
            t_gt = torch.from_numpy(patch_gt).permute(2, 0, 1).float()
            return t_lr, t_gt

        # Full resolution inference / validation
        t_lr = torch.from_numpy(norm_lr).permute(2, 0, 1).float()
        t_gt = torch.from_numpy(norm_gt).permute(2, 0, 1).float()
        return t_lr, t_gt

SemiconDataset = FastSemiconDataset
