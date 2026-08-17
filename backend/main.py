import os
import sys
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup import paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference_service import RestorationService
from backend.preprocessing import load_image_bytes, array_to_base64_png, extract_image_metadata

# Global inference service instance
service: RestorationService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("=" * 75)
    print("INITIALIZING SILICONVISION SERVER (FASTAPI + PYTORCH)...")
    service = RestorationService.get_instance()
    print("=" * 75)
    yield
    print("--> SiliconVision server shutting down.")

app = FastAPI(
    title="SiliconVision API",
    description="AI-Powered Semiconductor Image Restoration API (KLA SemiCon AI Hackathon)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "SiliconVision",
        "tagline": "From Noise to Precision",
        "model_loaded": service is not None and service.model is not None,
        "device": service.device_name.upper() if service else "UNKNOWN",
        "version": "1.0.0"
    }

@app.get("/api/model-info")
async def model_info():
    if not service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    return service.get_info()

@app.get("/api/demo-samples")
async def get_demo_samples():
    """
    Returns preloaded representative semiconductor validation samples for instant 1-click testing.
    """
    demo_dir = os.path.join(PROJECT_ROOT, "backend", "demo_samples")
    samples = [
        {
            "id": "sample_01",
            "name": "Grating Line Array (Pitch 45nm)",
            "category": "Line/Space Tracks",
            "description": "Dense parallel conductive track arrays degraded by speckle noise and 2× optical blur.",
            "lr_file": "sample_01_lr.npy",
            "gt_file": "sample_01_gt.npy",
            "has_gt": True
        },
        {
            "id": "sample_02",
            "name": "Contact Hole Matrix",
            "category": "Vias & Interconnects",
            "description": "Sub-micron cylindrical contact vias subject to severe signal attenuation and edge spread.",
            "lr_file": "sample_02_lr.npy",
            "gt_file": "sample_02_gt.npy",
            "has_gt": True
        },
        {
            "id": "sample_03",
            "name": "FinFET Transistor Gate Tracks",
            "category": "3D Transistor Logic",
            "description": "Complex nanostructure logic gate geometries requiring high-frequency spectral phase recovery.",
            "lr_file": "sample_03_lr.npy",
            "gt_file": "sample_03_gt.npy",
            "has_gt": True
        }
    ]

    for s in samples:
        lr_path = os.path.join(demo_dir, s["lr_file"])
        gt_path = os.path.join(demo_dir, s["gt_file"])
        if os.path.exists(lr_path):
            arr_lr = np.load(lr_path)
            s["lr_preview"] = array_to_base64_png(arr_lr, is_gt_or_restored=False)
            s["lr_metadata"] = extract_image_metadata(arr_lr, s["lr_file"])
        if os.path.exists(gt_path):
            arr_gt = np.load(gt_path)
            s["gt_preview"] = array_to_base64_png(arr_gt, is_gt_or_restored=True)

    return {"samples": samples}

@app.get("/api/demo-sample/{sample_id}")
async def load_demo_sample(sample_id: str):
    """
    Executes restoration on a selected demo sample.
    """
    demo_dir = os.path.join(PROJECT_ROOT, "backend", "demo_samples")
    lr_path = os.path.join(demo_dir, f"{sample_id}_lr.npy")
    gt_path = os.path.join(demo_dir, f"{sample_id}_gt.npy")

    if not os.path.exists(lr_path):
        raise HTTPException(status_code=404, detail=f"Demo sample {sample_id} not found")

    raw_lr = np.load(lr_path)
    raw_gt = np.load(gt_path) if os.path.exists(gt_path) else None

    result = service.restore_image(raw_lr=raw_lr, filename=f"{sample_id}_lr.npy", gt_arr=raw_gt)
    return result

@app.post("/api/restore")
async def restore_uploaded_file(file: UploadFile = File(...)):
    """
    Restores an uploaded NoisyLR image (.npy or .png/.jpg).
    """
    if not service:
        raise HTTPException(status_code=503, detail="Restoration service unavailable")

    allowed_exts = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{file.filename}'. Supported formats: .npy, .png, .jpg, .tif"
        )

    try:
        contents = await file.read()
        raw_arr = load_image_bytes(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image data: {str(e)}")

    if raw_arr.shape != (128, 128):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 128×128 input image, but received shape {raw_arr.shape}."
        )

    try:
        result = service.restore_image(raw_lr=raw_arr, filename=file.filename, gt_arr=None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restoration inference failed: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_pair(noisy_lr: UploadFile = File(...), ground_truth: UploadFile = File(...)):
    """
    Restores NoisyLR and computes exact PSNR and SSIM against the matching Ground Truth (GT).
    """
    if not service:
        raise HTTPException(status_code=503, detail="Restoration service unavailable")

    try:
        lr_bytes = await noisy_lr.read()
        gt_bytes = await ground_truth.read()

        raw_lr = load_image_bytes(lr_bytes, noisy_lr.filename)
        raw_gt = load_image_bytes(gt_bytes, ground_truth.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image pair: {str(e)}")

    if raw_lr.shape != (128, 128):
        raise HTTPException(status_code=400, detail=f"Expected 128×128 NoisyLR, got {raw_lr.shape}")
    if raw_gt.shape != (256, 256):
        raise HTTPException(status_code=400, detail=f"Expected 256×256 Ground Truth, got {raw_gt.shape}")

    try:
        result = service.restore_image(raw_lr=raw_lr, filename=noisy_lr.filename, gt_arr=raw_gt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline failed: {str(e)}")

@app.get("/api/download/{file_id}")
async def download_file(file_id: str, format: str = "png"):
    """
    Downloads cached restored output as .npy or .png.
    """
    ext = "npy" if format.lower() == "npy" else "png"
    target_path = os.path.join(PROJECT_ROOT, "outputs", f"restored_{file_id}.{ext}")

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Requested file not found or expired")

    media_type = "application/octet-stream" if ext == "npy" else "image/png"
    return FileResponse(
        target_path,
        media_type=media_type,
        filename=f"SiliconVision_restored_{file_id}.{ext}"
    )

# Mount static frontend
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
