import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64
from typing import Optional, Tuple

class ExplainableAI:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    def make_gradcam_heatmap(self, img_array: np.ndarray, 
                            pred_index: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        model = self.model_manager.get_current_model()
        if not model:
            raise ValueError("No model loaded")
        
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
            # Check base model if using transfer learning
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer = sublayer
                        break
                if last_conv_layer:
                    break
        
        if not last_conv_layer:
            # Fallback - return empty heatmap
            return np.zeros((224, 224))
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap /= tf.math.reduce_max(heatmap) + 1e-8
        
        return heatmap.numpy()
    
    def generate_heatmap_overlay(self, image: Image.Image, 
                                img_array: np.ndarray,
                                pred_index: int) -> str:
        """Generate heatmap overlay and return as base64"""
        try:
            # Generate heatmap
            heatmap = self.make_gradcam_heatmap(img_array, pred_index)
            
            # Resize heatmap to match image
            heatmap = cv2.resize(heatmap, (image.width, image.height))
            
            # Convert to colormap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), 
                cv2.COLORMAP_JET
            )
            
            # Convert PIL to CV2 format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create overlay
            overlay = cv2.addWeighted(img_cv, 0.7, heatmap_colored, 0.3, 0)
            
            # Convert back to RGB
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            # Convert to base64
            pil_image = Image.fromarray(overlay_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Return as data URL
            base64_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{base64_str}"
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            # Return a placeholder if error
            return ""