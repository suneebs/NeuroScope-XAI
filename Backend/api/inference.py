from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from core.model_manager import model_manager
from core.inference import InferenceEngine
from core.explainable_ai import ExplainableAI
import numpy as np

router = APIRouter()

# Initialize engines
inference_engine = InferenceEngine(model_manager)
xai_engine = ExplainableAI(model_manager)

@router.post("/infer")
async def infer(file: UploadFile = File(...)):
    """Run inference on uploaded image"""
    # Check if model is loaded
    if not model_manager.is_model_ready():
        raise HTTPException(400, "No model loaded. Please load a model first.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run inference
        results = inference_engine.predict(image)
        
        # Generate explainable AI visualization
        img_array = inference_engine.preprocess_image(image)
        pred_idx = model_manager.class_names.index(results['prediction']['label'])
        heatmap_overlay = xai_engine.generate_heatmap_overlay(image, img_array, pred_idx)
        
        # Add heatmap to results
        results['heatmap'] = heatmap_overlay
        
        return results
        
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

@router.post("/train")
async def start_training(config: dict):
    """Mock training endpoint"""
    # In production, implement actual training logic
    # For now, return a job ID
    import time
    job_id = f"job-{int(time.time())}"
    return {"jobId": job_id}

@router.get("/train/{job_id}/status")
async def training_status(job_id: str):
    """Mock training status endpoint"""
    # Simulate training progress
    import random
    progress = random.randint(0, 100)
    epoch = min(progress // 5, 19)
    
    return {
        "jobId": job_id,
        "status": "completed" if progress >= 100 else "running",
        "progress": progress,
        "epoch": epoch,
        "totalEpochs": 20,
        "metrics": {
            "loss": max(0.1, 1.5 - epoch * 0.07),
            "valLoss": max(0.12, 1.6 - epoch * 0.075)
        }
    }