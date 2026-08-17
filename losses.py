import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class CharbonnierLoss(nn.Module):
    """
    Smooth L1 Reconstruction Loss for sharp edge recovery and noise robustness.
    L(x, y) = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))

class SobelEdgeLoss(nn.Module):
    """
    Gradient Alignment Loss using Sobel filters to preserve fine semiconductor track boundaries.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)
        return loss_x + loss_y

class FFTLoss(nn.Module):
    """
    2D Frequency Domain Spectral Loss to penalize spectral energy discrepancies.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")

        mag_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        phase_loss = F.l1_loss(torch.angle(pred_fft), torch.angle(target_fft))
        return mag_loss + 0.1 * phase_loss

class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    """
    def __init__(self, window_size: int = 11, in_channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.channel = in_channels
        
        # Create Gaussian Window
        sigma = 1.5
        gauss = torch.tensor(
            [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
            dtype=torch.float32
        )
        gauss = (gauss / gauss.sum()).unsqueeze(1)
        _2d_window = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)
        self.window = nn.Parameter(_2d_window.expand(in_channels, 1, window_size, window_size).contiguous(), requires_grad=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()

class CompositeRestorationLoss(nn.Module):
    """
    Modular Composite Restoration Loss.
    Combines pixel, structural, edge, and spectral losses with configurable weights.
    """
    def __init__(
        self,
        w_pixel: float = 1.0,
        w_ssim: float = 0.5,
        w_fft: float = 0.1,
        w_sobel: float = 0.2,
        in_channels: int = 1
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.w_sobel = w_sobel

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(in_channels=in_channels)
        self.fft = FFTLoss()
        self.sobel = SobelEdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l_pix = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_fft = self.fft(pred, target)
        l_sobel = self.sobel(pred, target)

        total_loss = (
            self.w_pixel * l_pix +
            self.w_ssim * l_ssim +
            self.w_fft * l_fft +
            self.w_sobel * l_sobel
        )

        return {
            "total_loss": total_loss,
            "loss_pixel": l_pix.item(),
            "loss_ssim": l_ssim.item(),
            "loss_fft": l_fft.item(),
            "loss_sobel": l_sobel.item()
        }
