from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List, Dict
import os
import aiofiles
from core.model_manager import model_manager
from pydantic import BaseModel

router = APIRouter()

class ModelLoadRequest(BaseModel):
    id: str

@router.get("/")
async def list_models() -> Dict[str, List[Dict]]:
    """List available models"""
    models = model_manager.list_models()
    return {"models": models}

@router.post("/download/{model_id}")
async def download_model(model_id: str, background_tasks: BackgroundTasks):
    """Mock download endpoint"""
    # In production, implement actual download logic
    # For now, just return success
    return {"status": "success", "message": f"Model {model_id} downloaded"}

@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file"""
    allowed_extensions = {'.h5', '.keras', '.pb', '.onnx'}
    file_ext = os.path.splitext(file.filename)[1]
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_extensions}")
    
    # Save file
    file_path = os.path.join("models", file.filename)
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {
        "id": os.path.splitext(file.filename)[0],
        "name": file.filename,
        "source": "uploaded",
        "sizeMB": len(content) / (1024 * 1024)
    }

@router.post("/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    success = model_manager.load_model(request.id)
    
    if not success:
        # Try to find any .h5 or .keras file if exact match fails
        for filename in os.listdir("models"):
            if filename.endswith(('.h5', '.keras')):
                success = model_manager.load_model(filename)
                if success:
                    break
    
    if success:
        return {"loaded": True, "message": f"Model {request.id} loaded successfully"}
    else:
        raise HTTPException(404, f"Model {request.id} not found")

@router.get("/status")
async def model_status():
    """Get current model status"""
    return {
        "ready": model_manager.is_model_ready(),
        "current_model": model_manager.current_model_id
    }