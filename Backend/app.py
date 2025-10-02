import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"; os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, Dict
import json
from xai_true import TrueExplainableAIWeb # Assuming your XAI class is in this file

# --- Configuration ---
MODEL_DIR = "models"
DEFAULT_MODEL_SUBDIR = "bt_efficientnet_b4_v1" # <<< CHANGE THIS to your best model's folder name
# --- End Configuration ---

model = None
xai = None
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
last_load_error = ""

def load_model_from_path(model_subdir: str):
    """Loads a model from a specific subdirectory within MODEL_DIR."""
    global last_load_error
    
    model_path = os.path.join(MODEL_DIR, model_subdir, "model.keras")
    classes_path = os.path.join(MODEL_DIR, model_subdir, "classes.json")
    
    if not os.path.exists(model_path):
        last_load_error = f"Model file not found: {model_path}"
        print(f"ERROR: {last_load_error}")
        return None, None

    # Load class names
    loaded_classes = class_names # fallback
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r") as f:
                loaded_classes = json.load(f)
            print(f"✅ Loaded class names from {classes_path}")
        except Exception as e:
            print(f"⚠️  Could not load {classes_path}, using default class names. Error: {e}")
    else:
        print(f"⚠️  classes.json not found in {model_subdir}, using default class names.")

    # Load the Keras model
    try:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Successfully loaded model from {model_path}")
        last_load_error = ""
        return loaded_model, loaded_classes
    except Exception as e:
        last_load_error = f"Error loading model {model_path}: {e}"
        print(f"ERROR: {last_load_error}")
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """This function runs once when the server starts."""
    global model, xai, class_names
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Set TF to be deterministic
    import random
    SEED = 123
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    print("--- Server Startup ---")
    print(f"Attempting to load default model from: {DEFAULT_MODEL_SUBDIR}")
    
    loaded_model, loaded_classes = load_model_from_path(DEFAULT_MODEL_SUBDIR)
    
    if loaded_model and loaded_classes:
        model = loaded_model
        class_names = loaded_classes
        xai = TrueExplainableAIWeb(model, class_names, seed=SEED)
        print("✅ Model and XAI Analyzer are ready.")
    else:
        print("⚠️ WARNING: Model not loaded. Inference will fail. Place your trained model in the correct directory.")
        
    yield # The application runs here
    
    print("--- Server Shutdown ---")


app = FastAPI(title="NeuroScope XAI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    """Check if the backend model is loaded and ready."""
    return {
        "model_loaded": model is not None,
        "loaded_model_dir": DEFAULT_MODEL_SUBDIR if model else None,
        "class_names": class_names if model else None,
        "last_load_error": last_load_error,
    }

@app.post("/api/infer")
async def infer(file: UploadFile = File(...), mode: str = Query("full", enum=["quick", "full"])):
    """The main endpoint for running predictions."""
    if model is None or xai is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded on the server.", "hint": f"Ensure the '{DEFAULT_MODEL_SUBDIR}' folder with model.keras exists in '{MODEL_DIR}'."}
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)

        img_array = np.array(image_resized).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array).astype(np.float32)

        preds = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][predicted_idx])
        predicted_label = class_names[predicted_idx]

        xai_res = xai.explain_from_array(img_array, predicted_idx, predicted_label, confidence, mode=mode)
        
        # This structure should match your frontend component
        return JSONResponse({
            "prediction": {"label": predicted_label, "probability": confidence},
            "all_predictions": {cls: float(p) for cls, p in zip(class_names, preds[0])},
            "visualizations": xai_res["visualizations"],
            "interpretation": xai_res["interpretation"],
            "validation": xai_res["validation_result"],
            "summary": xai_res["summary"],
            "regions": xai_res["regions_analysis"],
            "regional_analysis": xai_res["regional_analysis"],
        })

    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"An error occurred during inference: {str(e)}")


if __name__ == "__main__":
    print("Starting NeuroScope XAI Backend...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)