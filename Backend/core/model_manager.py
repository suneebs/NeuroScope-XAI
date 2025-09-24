import os
import json
import tensorflow as tf
from typing import Dict, List, Optional
from datetime import datetime

class ModelManager:
    def __init__(self):
        self.models_dir = "models"
        self.loaded_model = None
        self.current_model_id = None
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
    def list_models(self) -> List[Dict]:
        """List available models"""
        models = []
        
        # Check for saved models
        for filename in os.listdir(self.models_dir):
            if filename.endswith(('.h5', '.keras')):
                model_path = os.path.join(self.models_dir, filename)
                model_info = {
                    "id": filename.replace('.h5', '').replace('.keras', ''),
                    "name": "BrainTumorNet" if "brain_tumor" in filename else filename,
                    "version": "1.0",
                    "sizeMB": round(os.path.getsize(model_path) / (1024 * 1024), 2),
                    "source": "local",
                    "created": datetime.fromtimestamp(os.path.getctime(model_path)).isoformat()
                }
                models.append(model_info)
        
        # Add downloadable models (mock)
        if not models:
            models = [
                {
                    "id": "bt-efficientnet-v1",
                    "name": "BrainTumorNet EfficientNet",
                    "version": "1.0",
                    "sizeMB": 45,
                    "source": "remote"
                },
                {
                    "id": "bt-efficientnet-xai",
                    "name": "BrainTumorNet XAI Enhanced",
                    "version": "1.1",
                    "sizeMB": 52,
                    "source": "remote"
                }
            ]
        
        return models
    
    def load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        try:
            # Try different file formats
            model_path = None
            for ext in ['.h5', '.keras']:
                path = os.path.join(self.models_dir, f"{model_id}{ext}")
                if os.path.exists(path):
                    model_path = path
                    break
            
            # For demo, also check exact filename
            if not model_path:
                path = os.path.join(self.models_dir, model_id)
                if os.path.exists(path):
                    model_path = path
            
            if model_path:
                self.loaded_model = tf.keras.models.load_model(model_path)
                self.current_model_id = model_id
                return True
            
            # If model doesn't exist locally, return False
            # In production, you might download it here
            return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_current_model(self):
        """Get currently loaded model"""
        return self.loaded_model
    
    def is_model_ready(self) -> bool:
        """Check if a model is loaded"""
        return self.loaded_model is not None

# Singleton instance
model_manager = ModelManager()