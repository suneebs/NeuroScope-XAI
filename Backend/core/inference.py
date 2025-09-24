import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, Tuple
import io
import base64

class InferenceEngine:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.input_size = (224, 224)
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize
        image = image.resize(self.input_size)
        
        # Convert to array
        img_array = np.array(image)
        
        # Ensure 3 channels
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for EfficientNet
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image: Image.Image) -> Dict:
        """Run inference on image"""
        model = self.model_manager.get_current_model()
        if not model:
            raise ValueError("No model loaded")
        
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Get all predictions
        results = []
        for idx, prob in enumerate(predictions[0]):
            results.append({
                "label": self.model_manager.class_names[idx],
                "probability": float(prob)
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            "prediction": {
                "label": self.model_manager.class_names[predicted_idx],
                "probability": confidence
            },
            "topK": results,
            "raw_predictions": predictions[0].tolist()
        }