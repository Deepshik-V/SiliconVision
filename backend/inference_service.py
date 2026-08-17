import os
import sys
import time
import uuid
import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import BaselineSemiconNet
from normalization import get_scaler
from backend.preprocessing import array_to_base64_png, extract_image_metadata
from backend.metrics import evaluate_restoration

class RestorationService:
    _instance = None

    def __init__(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
        
        self.checkpoint_path = checkpoint_path
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        
        self.model_name = "BaselineSemiconNet (NAFNet + 2D FFT Spectral Attention)"
        self.scale_factor = 2
        self.in_channels = 1
        self.out_channels = 1
        self.width = 32
        self.scaler = get_scaler("per_image")
        
        self.output_cache_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(self.output_cache_dir, exist_ok=True)
        
        self.model = None
        self.param_count = 0
        self.load_model()

    @classmethod
    def get_instance(cls, checkpoint_path: str = None):
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def load_model(self):
        print(f"--> [RestorationService] Initializing {self.model_name} on device: {self.device_name.upper()}...")
        start_t = time.time()
        
        self.model = BaselineSemiconNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            scale_factor=self.scale_factor
        ).to(self.device)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())
        load_duration = (time.time() - start_t) * 1000.0
        print(f"[+] [RestorationService] Loaded model ({self.param_count:,} params) in {load_duration:.1f}ms.")

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "architecture": "NAFNet-SR + Frequency Spectral Attention",
            "parameter_count": self.param_count,
            "parameter_count_formatted": f"{self.param_count / 1e6:.2f}M",
            "scale_factor": f"{self.scale_factor}× Super-Resolution",
            "input_resolution": "128 × 128",
            "output_resolution": "256 × 256",
            "device": self.device_name.upper(),
            "checkpoint_path": os.path.basename(self.checkpoint_path),
            "status": "Ready / Online"
        }

    def restore_image(self, raw_lr: np.ndarray, filename: str = "input.npy", gt_arr: np.ndarray = None) -> dict:
        """
        Executes full restoration pipeline with verified telemetry.
        """
        assert raw_lr.ndim == 2, f"Expected 2D image, got shape: {raw_lr.shape}"
        assert raw_lr.shape == (128, 128), f"Expected (128, 128) input, got {raw_lr.shape}"

        pipeline_timeline = {}
        t_start = time.time()

        # Step 1: Input Ingestion & Metadata
        input_meta = extract_image_metadata(raw_lr, filename)
        input_b64 = array_to_base64_png(raw_lr, is_gt_or_restored=False)
        pipeline_timeline["ingestion_ms"] = round((time.time() - t_start) * 1000.0, 2)

        # Step 2: Normalization Preprocessing
        t_prep_start = time.time()
        raw_lr_exp = np.expand_dims(raw_lr, -1)
        norm_lr = self.scaler(raw_lr_exp)
        img_tensor = torch.from_numpy(norm_lr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        pipeline_timeline["preprocessing_ms"] = round((time.time() - t_prep_start) * 1000.0, 2)

        # Step 3: Neural Restoration Inference
        t_infer_start = time.time()
        with torch.inference_mode():
            pred_tensor = self.model(img_tensor)
        inference_duration_ms = (time.time() - t_infer_start) * 1000.0
        pipeline_timeline["neural_inference_ms"] = round(inference_duration_ms, 2)

        # Step 4: Post-Processing & Output Validation
        t_post_start = time.time()
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
        pred_np = pred_tensor.squeeze().cpu().numpy().astype(np.float32)

        assert pred_np.shape == (256, 256), f"Output shape invalid: {pred_np.shape}"
        assert np.isfinite(pred_np).all(), "NaN or Inf detected in restored output!"
        assert pred_np.min() >= 0.0 and pred_np.max() <= 1.0, f"Value range out of [0, 1]: [{pred_np.min()}, {pred_np.max()}]"
        pipeline_timeline["postprocessing_ms"] = round((time.time() - t_post_start) * 1000.0, 2)

        # Step 5: Output Previews & Caching for Download
        file_id = str(uuid.uuid4())[:8]
        output_npy_path = os.path.join(self.output_cache_dir, f"restored_{file_id}.npy")
        output_png_path = os.path.join(self.output_cache_dir, f"restored_{file_id}.png")

        np.save(output_npy_path, pred_np)
        
        # Save displayable 8-bit PNG
        pil_out = Image.fromarray((pred_np * 255.0).astype(np.uint8), mode="L")
        pil_out.save(output_png_path)

        output_meta = extract_image_metadata(pred_np, f"restored_{file_id}.npy")
        output_b64 = array_to_base64_png(pred_np, is_gt_or_restored=True)

        total_duration_ms = (time.time() - t_start) * 1000.0

        # Step 6: Evaluation Metrics if GT is provided
        eval_metrics = None
        gt_b64 = None
        if gt_arr is not None:
            if gt_arr.shape == (256, 256):
                eval_metrics = evaluate_restoration(pred_np, gt_arr)
                gt_b64 = array_to_base64_png(gt_arr, is_gt_or_restored=True)

        return {
            "file_id": file_id,
            "status": "Success",
            "input_metadata": input_meta,
            "output_metadata": output_meta,
            "input_preview": input_b64,
            "output_preview": output_b64,
            "gt_preview": gt_b64,
            "evaluation_metrics": eval_metrics,
            "latency": {
                "inference_ms": round(inference_duration_ms, 2),
                "total_pipeline_ms": round(total_duration_ms, 2),
                "fps": round(1000.0 / max(1.0, inference_duration_ms), 1),
                "timeline": pipeline_timeline
            },
            "device": self.device_name.upper(),
            "model_info": self.get_info(),
            "download_urls": {
                "npy": f"/api/download/{file_id}?format=npy",
                "png": f"/api/download/{file_id}?format=png"
            }
        }
