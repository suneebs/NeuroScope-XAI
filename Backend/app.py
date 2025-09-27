import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"; os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_CUDNN_DETERMINISTIC"] = "1"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any
import json
from training_manager import TrainingManager
from xai_true import TrueExplainableAIWeb

MODEL_DIR = "models"
model = None; xai = None; last_load_error: str = ""
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
SEED = 123
TRAINER = TrainingManager(models_dir="models", seed=SEED)

def list_model_files() -> Dict[str, List[Dict]]:
    items = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, f)
            if os.path.isfile(path):
                items.append({"filename": f, "sizeMB": round(os.path.getsize(path) / (1024 * 1024), 2), "mtime": os.path.getmtime(path)})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"models": items}

def try_load_keras_model(path: str):
    global last_load_error
    try:
        print(f"-> Loading Keras model: {path}")
        m = tf.keras.models.load_model(path, compile=False)
        last_load_error = ""; return m
    except Exception as e:
        last_load_error = f"Keras load_model failed for {path}: {e}"; print("   Failed:", last_load_error); return None

def try_rebuild_and_load_weights(weights_path: str, num_classes: int = 4):
    from tensorflow.keras import Sequential; from tensorflow.keras.layers import Dense, Dropout
    global last_load_error
    try:
        print(f"-> Rebuilding architecture and loading weights from: {weights_path}")
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights=None, input_shape=(224,224,3), pooling='max')
        m = Sequential([base, Dense(256, activation='relu'), Dropout(rate=0.45, seed=123), Dense(num_classes, activation='softmax')])
        m.load_weights(weights_path, by_name=True, skip_mismatch=True)
        last_load_error = ""; return m
    except Exception as e:
        last_load_error = f"Rebuild+load_weights failed for {weights_path}: {e}"; print(f"   Failed: {last_load_error}"); return None

def load_model_from_dir(prefer_filename: Optional[str] = None):
    files = list_model_files()["models"]
    if prefer_filename:
        cand = os.path.join(MODEL_DIR, prefer_filename)
        if os.path.isfile(cand) and prefer_filename.lower().endswith((".h5",".keras")):
            m = try_load_keras_model(cand) or try_rebuild_and_load_weights(cand, num_classes=len(class_names))
            if m: return m
    keras_files = [os.path.join(MODEL_DIR, f["filename"]) for f in files if f["filename"].lower().endswith((".h5",".keras"))]
    for path in keras_files:
        m = try_load_keras_model(path) or try_rebuild_and_load_weights(path, num_classes=len(class_names))
        if m: return m
    print("No loadable model found."); return None

def load_classes(path: Optional[str] = None):
    if not path: path = os.path.join(MODEL_DIR, "classes.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f: arr = json.load(f)
            if isinstance(arr, list) and len(arr) == 4: return arr
        except Exception as e: print("Failed to load classes.json:", e)
    return ["glioma","meningioma","no_tumor","pituitary"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, xai, class_names, last_load_error
    os.makedirs(MODEL_DIR, exist_ok=True)
    import random; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    mpath, cpath = TRAINER.get_active_artifacts()
    loaded = False
    if mpath and os.path.exists(mpath):
        class_names = load_classes(cpath)
        m = try_load_keras_model(mpath) or try_rebuild_and_load_weights(mpath, num_classes=len(class_names))
        if m: model = m; xai = TrueExplainableAIWeb(model, class_names, seed=SEED); last_load_error = ""; print("✅ Loaded trained model:", mpath); loaded = True
    if not loaded:
        class_names = load_classes()
        m = load_model_from_dir()
        if m: model = m; xai = TrueExplainableAIWeb(model, class_names, seed=SEED); last_load_error = ""; print("✅ Loaded model from models/")
        else: print("⚠️ Model not loaded. Use /api/models/reload or train a new one.")
    yield
    print("👋 Shutting down backend...")

app = FastAPI(title="NeuroScope XAI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root(): return {"message": "NeuroScope XAI API", "status": "running", "model_loaded": model is not None}
@app.get("/api/status")
async def status(): return {"model_loaded": model is not None, "contents": os.listdir(MODEL_DIR), "last_load_error": last_load_error, "class_names": class_names}
@app.get("/api/models")
async def models_list(): return list_model_files()

@app.post("/api/models/reload")
async def models_reload(filename: Optional[str] = Query(None)):
    global model, xai, last_load_error
    m = load_model_from_dir(prefer_filename=filename)
    if not m: return {"loaded": False, "message": "No loadable model found", "last_load_error": last_load_error}
    model = m; xai = TrueExplainableAIWeb(model, class_names, seed=SEED); last_load_error = ""; return {"loaded": True}

@app.post("/api/models/use")
async def models_use(payload: Dict[str, str]):
    global model, xai, last_load_error, class_names
    modelPath = payload.get("modelPath"); classesPath = payload.get("classesPath")
    if not modelPath or not os.path.exists(modelPath): raise HTTPException(404, f"Model not found: {modelPath}")
    class_names = load_classes(classesPath)
    m = try_load_keras_model(modelPath) or try_rebuild_and_load_weights(modelPath, num_classes=len(class_names))
    if not m: raise HTTPException(500, "Could not load model")
    model = m; xai = TrueExplainableAIWeb(model, class_names, seed=SEED); last_load_error = ""; return {"active": True}

@app.get("/api/infer")
async def infer_help(): return {"hint": "Use POST with multipart/form-data. Example: curl -F \"file=@path.jpg\" /api/infer"}
@app.post("/api/infer")
async def infer(file: UploadFile = File(...), mode: str = Query("full", enum=["quick","full"])):
    if not model or not xai: return JSONResponse({"error": "Model not loaded"}, status_code=503)
    try:
        contents = await file.read(); image = Image.open(io.BytesIO(contents)).convert("RGB"); image_resized = image.resize((224, 224), resample=Image.BILINEAR)
        img_array = np.array(image_resized).astype(np.float32); img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array).astype(np.float32)
        preds = model.predict(img_array, verbose=0); predicted_idx = int(np.argmax(preds[0])); confidence = float(preds[0][predicted_idx]); predicted_label = class_names[predicted_idx]
        xai_res = xai.explain_from_array(img_array, predicted_idx, predicted_label, confidence, mode=mode)
        return JSONResponse({"prediction": {"label": predicted_label, "probability": confidence}, "all_predictions": {cls: float(p) for cls, p in zip(class_names, preds[0])}, "visualizations": xai_res["visualizations"], "interpretation": xai_res["interpretation"], "validation": xai_res["validation_result"], "summary": xai_res["summary"], "regions": xai_res["regions_analysis"], "regional_analysis": xai_res["regional_analysis"], "analysis_metadata": {"model_type": "EfficientNetB4", "analysis_time": xai_res["summary"]["analysis_time"], "methods_used": ["Occlusion Analysis", "Pattern Recognition"], "disclaimer": "This AI analysis is for screening purposes only."}})
    except Exception as e: print("Inference error:", e); raise HTTPException(500, f"Inference error: {str(e)}")

@app.post("/api/train/start")
async def train_start(payload: Dict[str, Any]): return {"jobId": TRAINER.start_training(payload)}
@app.get("/api/train/status")
async def train_status(jobId: str):
    st = TRAINER.get_status(jobId)
    if not st: raise HTTPException(404, "job not found")
    return {"jobId": st.job_id, "status": st.status, "message": st.message, "phase": st.phase, "epoch": st.epoch, "totalEpochs": st.total_epochs, "progress": st.progress, "metrics": st.metrics_history[-1] if st.metrics_history else {}, "artifacts": st.artifacts, "startedAt": st.started_at, "endedAt": st.ended_at}
@app.post("/api/train/stop")
async def train_stop(jobId: str): return {"stopped": TRAINER.stop(jobId)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)