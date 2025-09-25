# backend/app.py (full file for clarity)
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, List, Tuple, Dict

from xai_true import TrueExplainableAIWeb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

MODEL_DIR = "models"
model = None
xai = None
last_load_error: str = ""
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

def list_model_files() -> Dict[str, List[Dict]]:
    items = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, f)
            if os.path.isfile(path):
                items.append({
                    "filename": f,
                    "sizeMB": round(os.path.getsize(path) / (1024 * 1024), 2),
                    "mtime": os.path.getmtime(path),
                })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"models": items}

def try_load_keras_model(path: str):
    global last_load_error
    try:
        print(f"-> Loading Keras model: {path}")
        m = tf.keras.models.load_model(path, compile=False)
        print("   Loaded OK (Keras model).")
        last_load_error = ""
        return m
    except Exception as e:
        last_load_error = f"Keras load_model failed for {path}: {e}"
        print("   Failed:", last_load_error)
        return None

def try_rebuild_and_load_weights(weights_path: str, num_classes: int = 4):
    # Minimal architecture rebuild for weight fallback
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    try:
        print(f"-> Rebuilding architecture and loading weights from: {weights_path}")
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights=None, input_shape=(224,224,3), pooling='max')
        m = Sequential([base, Dense(256, activation='relu'), Dropout(rate=0.45, seed=123), Dense(num_classes, activation='softmax')])
        m.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("   Weights loaded OK (by_name=True, skip_mismatch=True).")
        return m
    except Exception as e:
        print(f"   Rebuild+load_weights failed for {weights_path}: {e}")
        return None

def load_model_from_dir(prefer_filename: Optional[str] = None):
    files = list_model_files()["models"]

    # Specific filename
    if prefer_filename:
        cand = os.path.join(MODEL_DIR, prefer_filename)
        if os.path.isfile(cand) and prefer_filename.lower().endswith((".h5",".keras")):
            m = try_load_keras_model(cand)
            if m: return m
            m = try_rebuild_and_load_weights(cand, num_classes=len(class_names))
            if m: return m

    # Newest .h5/.keras
    keras_files = [os.path.join(MODEL_DIR, f["filename"]) for f in files if f["filename"].lower().endswith((".h5",".keras"))]
    for path in keras_files:
        m = try_load_keras_model(path)
        if m: return m

    # As weights fallback
    for path in keras_files:
        m = try_rebuild_and_load_weights(path, num_classes=len(class_names))
        if m: return m

    print("No loadable model found.")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, xai
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"🗂 CWD: {os.getcwd()}")
    print(f"📦 {MODEL_DIR}/: {os.listdir(MODEL_DIR)}")
    model = load_model_from_dir()
    if model:
        xai = TrueExplainableAIWeb(model, class_names)
        print("✅ XAI initialized")
    else:
        print("⚠️ Model not loaded. Use /api/models/reload")
    yield
    print("👋 Shutting down backend...")

app = FastAPI(title="NeuroScope XAI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "NeuroScope XAI API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {"status": "/api/status", "models": "/api/models", "inference": "/api/infer", "docs": "/docs"},
    }

@app.get("/api/status")
async def status():
    return {
        "model_loaded": model is not None,
        "models_dir": MODEL_DIR,
        "contents": os.listdir(MODEL_DIR),
        "last_load_error": last_load_error,
    }

@app.get("/api/models")
async def models_list():
    return list_model_files()

@app.post("/api/models/reload")
async def models_reload(filename: Optional[str] = Query(None)):
    global model, xai
    m = load_model_from_dir(prefer_filename=filename)
    if not m:
        return {"loaded": False, "message": "No loadable model found", "last_load_error": last_load_error}
    model = m
    xai = TrueExplainableAIWeb(model, class_names)
    return {"loaded": True}

@app.post("/api/infer")
async def infer(file: UploadFile = File(...), mode: str = Query("quick", enum=["quick","full"])):
    if model is None or xai is None:
        return JSONResponse({"error": "Model not loaded", "hint": "POST /api/models/reload"}, status_code=503)
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_resized = image.resize((224, 224))

        # Ensure float32 for preprocess
        img_array = np.array(image_resized).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array).astype(np.float32)

        preds = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][predicted_idx])
        predicted_label = class_names[predicted_idx]

        xai_res = xai.explain_from_array(img_array, predicted_idx, predicted_label, confidence, mode=mode)

        return JSONResponse({
            "prediction": {"label": predicted_label, "probability": confidence},
            "all_predictions": {cls: float(p) for cls, p in zip(class_names, preds[0])},
            "visualizations": xai_res["visualizations"],
            "interpretation": xai_res["interpretation"],
            "validation": xai_res["validation_result"],
            "summary": xai_res["summary"],
            "regions": xai_res["regions_analysis"],
            "regional_analysis": xai_res["regional_analysis"],
            "analysis_metadata": {
                "model_type": "EfficientNetB4",
                "analysis_time": xai_res["summary"]["analysis_time"],
                "methods_used": ["Occlusion Analysis", "Pattern Recognition"],
                "disclaimer": "This AI analysis is for screening purposes only. Clinical correlation and professional medical evaluation are essential for diagnosis.",
            },
        })
    except Exception as e:
        print("Inference error:", e)
        raise HTTPException(500, f"Inference error: {str(e)}")

if __name__ == "__main__":
    print("🌐 http://localhost:8000  |  📚 http://localhost:8000/docs")
    print(f"🗂  CWD: {os.getcwd()}")
    if os.path.isdir(MODEL_DIR):
        print("📦 models/:", os.listdir(MODEL_DIR))
    else:
        print("📦 models/: <missing>")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)