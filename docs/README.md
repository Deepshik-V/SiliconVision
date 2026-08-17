# SiliconVision Documentation & Presentation Assets

This directory contains visual architecture diagrams, UI demo showcase screenshots, and presentation materials.

---

## Included Assets

1. **`architecture.png`**: High-resolution system architecture diagram detailing:
   * $128 \times 128 \text{ NoisyLR}$ Sensor Ingestion
   * Quantile Dynamic Calibration & Preprocessing
   * 4-Stage NAFNet Backbone with 2D FFT Frequency Spectral Attention
   * $2\times$ PixelShuffle Super-Resolution Reconstruction Head
   * Composite Loss Formulation ($\mathcal{L}_{\text{Charbonnier}} + \mathcal{L}_{\text{SSIM}} + \mathcal{L}_{\text{FFT}} + \mathcal{L}_{\text{Sobel}}$)

2. **`demo-screenshot.png`**: Visual demonstration screenshot showing:
   * Input Inspection Panel with statistical telemetry & speckle range indicators
   * Animated 5-stage processing pipeline telemetry
   * High-resolution restoration comparison viewer

3. **`presentation.pdf`**:
   * *Status*: Place your finalized hackathon presentation slide deck here (`docs/presentation.pdf`) when preparing for judging submissions.
