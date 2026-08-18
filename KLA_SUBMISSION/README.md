# SiliconVision — KLA SemiCon AI Hackathon Submission

> **Team Solution**: SiliconVision  
> **Problem Statement**: Semiconductor Image Restoration & 2× Super-Resolution Reconstruction (PS-01)  
> **Input**: Degraded Low-Resolution Semiconductor Image (`128 × 128`, `.npy`)  
> **Output**: Clean High-Resolution Restored Image (`256 × 256`, `.npy`, `float32`, range `[0.0, 1.0]`)  

---

## 1. Problem Description

Optical semiconductor defect metrology systems suffer from severe sensor degradations:
- **Speckle Noise Bursts**: Multiplicative coherent noise with out-of-bound dynamic range values.
- **Additive Gaussian Sensor Noise**: Thermal readout noise obscuring fine critical dimension edges.
- **Optical Diffraction & Sub-sampling**: Loss of high-frequency line/space grating patterns ($128 \times 128 \text{ NoisyLR} \to 256 \times 256 \text{ GT}$).

**Goal**: Restore fine nanofabrication structures, suppress noise, and upscale spatial resolution by $2\times$.

---

## 2. Model Architecture

* **Architecture**: **NAFNet Backbone + 2D FFT Spectral Attention + 2× PixelShuffle Head**
* **Parameter Count**: **18,211,009 parameters (18.21M)**
* **Backbone**: Non-linear Activation Free gating (`SimpleGate`) with Simplified Channel Attention (`SCA`).
* **Frequency Modulation**: 2D Fast Fourier Transform (`rfft2`) spectral feature modulation block (`HighFrequencyFourierAttention`) restoring periodic semiconductor line arrays.
* **Super-Resolution Reconstruction**: Sub-pixel convolution (`PixelShuffle(2)`) upscaling from $128 \times 128 \to 256 \times 256$.
* **Trained Weights**: Self-contained in `models/best_model.pth`.

---

## 3. Environment & Dependencies

* **Python Version**: Python 3.10+
* **Core Libraries**:
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `numpy>=1.24.0`

### Installation
```bash
pip install -r requirements.txt
```

---

## 4. Execution Command

Run batch restoration on any input directory of `.npy` files using the standard official execution syntax:

```bash
python run.py <input-dir> <output-dir>
```

### Example
```bash
python run.py ./test_input ./test_output
```

---

## 5. Input & Output Specification

| Specification | Input (`NoisyLR`) | Output (`Restored`) |
| :--- | :--- | :--- |
| **File Format** | 2D NumPy Array (`.npy`) | 2D NumPy Array (`.npy`) |
| **Spatial Dimensions** | $128 \times 128$ | $256 \times 256$ ($2\times$ Super-Resolution) |
| **Data Type** | `float32` | `float32` |
| **Value Range** | Dynamic (may contain speckle values $<0$ or $>1$) | Strictly clamped to $[0.0, 1.0]$ |
| **Data Integrity** | Grayscale, finite | **0 NaN**, **0 Inf** guaranteed |
| **Filename Mapping** | `filename.npy` | Exactly matches `filename.npy` |

---

## 6. Offline Execution & Hardware Compatibility

* **100% Offline Execution**: All weights and preprocessing code are self-contained in `models/`. The script makes **0 internet requests**, requires **0 API keys**, and needs **0 user interaction**.
* **Automatic Hardware Detection**:
  - **CUDA Available**: Automatically executes on CUDA GPU for maximum throughput.
  - **CPU Fallback**: Automatically and seamlessly falls back to multithreaded CPU execution when no GPU is present.

---

## 7. Submission Package Directory Structure

```
KLA_SUBMISSION/
├── run.py                 # Official execution script: python run.py <input> <output>
├── requirements.txt       # Minimal required Python dependencies
├── README.md              # Technical documentation and execution guide
└── models/
    ├── __init__.py        # Module exports
    ├── model.py           # 18.21M parameter NAFNet + 2D FFT restoration architecture
    ├── normalization.py   # Self-contained quantile percentile preprocessing
    └── best_model.pth     # Trained PyTorch model weights (219 MB)
```
