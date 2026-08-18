"""
SiliconVision Neural Restoration Model Architecture
===================================================
Non-linear Activation Free Network (NAFNet) with 2D Fast Fourier Transform (FFT)
Spectral Attention Bottleneck and Sub-Pixel Convolution (PixelShuffle) 2x SR Head.

Total Parameters: 18,211,009 (18.21M)
Input: (B, 1, 128, 128) float32 NoisyLR
Output: (B, 1, 256, 256) float32 Restored
"""

import os
import torch
import torch.nn as nn
import torch.fft

class LayerNorm2d(nn.Module):
    """
    2D Channel-wise Layer Normalization with 3D parameters (C, 1, 1).
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels, 1, 1)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels, 1, 1)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight * y + self.bias

class SimpleGate(nn.Module):
    """
    Nonlinear Activation Free gating mechanism.
    Splits channel dimension in half and multiplies the two halves.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block with SimpleGate and Simplified Channel Attention.
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class HighFrequencyFourierAttention(nn.Module):
    """
    2D FFT Frequency Spectral Attention Module for High-Frequency Semiconductor Feature Recovery.
    Matches exact trained checkpoint weight keys: conv_fourier.0 and conv_fourier.2.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv_fourier = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm="backward")
        real = fft_x.real
        imag = fft_x.imag
        freq_feat = torch.cat([real, imag], dim=1)
        
        weight = self.conv_fourier(freq_feat)
        w_real, w_imag = weight.chunk(2, dim=1)
        
        real_scaled = real * w_real
        imag_scaled = imag * w_imag
        fft_scaled = torch.complex(real_scaled, imag_scaled)
        
        out = torch.fft.irfft2(fft_scaled, s=(H, W), norm="backward")
        return x + out

class BaselineSemiconNet(nn.Module):
    """
    SiliconVision Deep Neural Network for Semiconductor Image Restoration.
    4-Stage Encoder-Decoder NAFNet with 2D FFT Spectral Attention and 2x PixelShuffle.
    
    Trained Checkpoint Compatibility:
    - Width: 32
    - Scale Factor: 2
    - Encoders: [2, 2, 4, 6]
    - Middle: 6 blocks + HighFrequencyFourierAttention
    - Decoders: [2, 2, 2, 2]
    - Parameters: 18,211,009
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        enc_blk_nums: list = None,
        middle_blk_num: int = 6,
        dec_blk_nums: list = None,
        scale_factor: int = 2
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 6]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.scale_factor = scale_factor
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=1, bias=True)

        # 4-Stage Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        # Bottleneck with 2D FFT Attention
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)],
            HighFrequencyFourierAttention(chan)
        )

        # 4-Stage Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=True),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # 2x Super-Resolution PixelShuffle Head
        self.sr_head = nn.Sequential(
            nn.Conv2d(width, width * (scale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        out = self.sr_head(x)
        return out

# Semantic alias
SiliconVisionRestorationNet = BaselineSemiconNet

def load_restoration_model(checkpoint_path: str = None, device: torch.device = None) -> nn.Module:
    """
    Loads and returns the trained restoration model ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {checkpoint_path}")

    model = BaselineSemiconNet(in_channels=1, out_channels=1, width=32, scale_factor=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
