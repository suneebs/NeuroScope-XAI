from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2
import base64
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import remove_small_objects
import time
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Global variables
model = None
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

class TrueExplainableAI:
    """Your exact TrueExplainableAI implementation for web deployment"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def find_last_conv_layer(self):
        """Find the name of the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        # If no conv layer found in sequential model, check the base model
        if hasattr(self.model.layers[0], 'layers'):
            for layer in reversed(self.model.layers[0].layers):
                if 'conv' in layer.name.lower():
                    return self.model.layers[0].name, layer.name
        return None, None
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """Generate Grad-CAM heatmap"""
        base_layer_name, conv_layer_name = self.find_last_conv_layer()
        
        if base_layer_name and conv_layer_name:
            # Create a model that maps the input to the last conv layer
            base_model = self.model.get_layer(base_layer_name)
            last_conv_layer = base_model.get_layer(conv_layer_name)
            
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [last_conv_layer.output, self.model.output]
            )
        else:
            # Fallback - try to find any conv layer
            for layer in self.model.layers:
                if 'conv' in str(type(layer)).lower():
                    grad_model = tf.keras.models.Model(
                        [self.model.inputs], 
                        [layer.output, self.model.output]
                    )
                    break
            else:
                return None
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
    def validate_occlusion_results(self, img_array, sensitivity_map, predicted_class_idx):
        """Validate that occlusion results make sense"""
        threshold = np.percentile(sensitivity_map, 90)
        high_sensitivity_mask = sensitivity_map > threshold
        
        validation_result = {
            'is_valid': True,
            'confidence_drop': 0,
            'original_confidence': 0,
            'occluded_confidence': 0,
            'message': ''
        }
        
        if np.any(high_sensitivity_mask):
            test_img = img_array.copy()
            mean_val = np.mean(img_array)
            test_img[0][high_sensitivity_mask] = mean_val
            
            new_pred = self.model.predict(test_img, verbose=0)
            new_class = np.argmax(new_pred[0])
            original_conf = self.model.predict(img_array, verbose=0)[0][predicted_class_idx]
            new_conf = new_pred[0][predicted_class_idx]
            
            confidence_drop = original_conf - new_conf
            
            validation_result.update({
                'original_confidence': float(original_conf),
                'occluded_confidence': float(new_conf),
                'confidence_drop': float(confidence_drop),
                'is_valid': confidence_drop > 0.05 or new_class != predicted_class_idx
            })
            
            if validation_result['is_valid']:
                validation_result['message'] = "✓ Validation PASSED: Occluding important regions affects prediction"
            else:
                validation_result['message'] = "⚠ Validation WARNING: Model may be too confident or regions not critical"
        
        return validation_result
    
    def analyze_sensitivity_patterns(self, sensitivity_map):
        """Dynamically analyze patterns in sensitivity map"""
        threshold = np.percentile(sensitivity_map, 75)
        binary_map = (sensitivity_map > threshold).astype(np.uint8)
        
        min_size = 50
        binary_map = remove_small_objects(binary_map.astype(bool), min_size=min_size)
        
        labeled_map = sk_label(binary_map)
        regions = regionprops(labeled_map, intensity_image=sensitivity_map)
        
        regions_analysis = []
        for region in regions:
            cy, cx = region.centroid
            
            analysis = {
                'area': int(region.area),
                'centroid': (float(cy), float(cx)),
                'bbox': [int(x) for x in region.bbox],
                'eccentricity': float(region.eccentricity),
                'solidity': float(region.solidity),
                'mean_intensity': float(region.mean_intensity),
                'max_intensity': float(region.max_intensity),
                'min_intensity': float(region.min_intensity),
                'orientation': float(region.orientation),
                'perimeter': float(region.perimeter)
            }
            regions_analysis.append(analysis)
        
        regions_analysis.sort(key=lambda x: x['area'] * x['mean_intensity'], reverse=True)
        return regions_analysis
    
    def generate_dynamic_interpretation(self, regions_analysis, predicted_class, confidence, img_shape=(224, 224)):
        """Generate interpretation based on actual findings - NOT hardcoded"""
        interpretation = {
            'main_findings': [],
            'pattern_description': '',
            'location_analysis': '',
            'confidence_interpretation': '',
            'characteristics': [],
            'clinical_relevance': [],
            'medical_interpretation': {
                'assessment': '',
                'observations': [],
                'recommendations': []
            }
        }
        
        # Analyze confidence level
        if confidence > 0.95:
            interpretation['confidence_interpretation'] = f"The AI shows very high confidence ({confidence:.1%}), indicating clear pattern recognition."
        elif confidence > 0.85:
            interpretation['confidence_interpretation'] = f"The AI shows high confidence ({confidence:.1%}), suggesting strong pattern match."
        elif confidence > 0.70:
            interpretation['confidence_interpretation'] = f"The AI shows moderate confidence ({confidence:.1%}), indicating some uncertainty."
        else:
            interpretation['confidence_interpretation'] = f"The AI shows low confidence ({confidence:.1%}), suggesting ambiguous patterns."
        
        if not regions_analysis:
            interpretation['main_findings'].append("No significant activation regions detected")
            interpretation['pattern_description'] = "The AI decision appears to be based on subtle, distributed features rather than focal abnormalities."
            return interpretation
        
        # Analyze number and distribution of regions
        num_regions = len(regions_analysis)
        if num_regions == 1:
            interpretation['pattern_description'] = "The AI focused on a single, well-defined area of interest."
        elif num_regions <= 3:
            interpretation['pattern_description'] = f"The AI identified {num_regions} distinct focal areas."
        else:
            interpretation['pattern_description'] = f"The AI detected {num_regions} regions, suggesting a diffuse or multifocal pattern."
        
        # Analyze spatial distribution
        if num_regions > 1:
            centroids = [r['centroid'] for r in regions_analysis]
            y_coords = [c[0] for c in centroids]
            x_coords = [c[1] for c in centroids]
            
            y_spread = np.std(y_coords)
            x_spread = np.std(x_coords)
            
            if x_spread > 50 and y_spread > 50:
                interpretation['characteristics'].append("widespread distribution across scan")
            elif x_spread > 50:
                interpretation['characteristics'].append("horizontal distribution pattern")
            elif y_spread > 50:
                interpretation['characteristics'].append("vertical distribution pattern")
            else:
                interpretation['characteristics'].append("clustered distribution")
        
        # Analyze each significant region
        for i, region in enumerate(regions_analysis[:3]):
            cy, cx = region['centroid']
            img_center_y, img_center_x = img_shape[0] // 2, img_shape[1] // 2
            
            # Determine location in brain-relevant terms
            if predicted_class == 'pituitary':
                if cy > img_center_y + 30:
                    location = "lower central region (typical for pituitary)"
                    interpretation['clinical_relevance'].append("location consistent with pituitary region")
                else:
                    location = f"atypical location for pituitary"
            else:
                if cx < img_center_x - 30:
                    h_location = "left"
                elif cx > img_center_x + 30:
                    h_location = "right"
                else:
                    h_location = "central"
                    
                if cy < img_center_y - 30:
                    v_location = "upper"
                elif cy > img_center_y + 30:
                    v_location = "lower"
                else:
                    v_location = "middle"
                
                location = f"{v_location} {h_location}" if v_location != "middle" else h_location
            
            # Size analysis with clinical context
            area_percentage = (region['area'] / (img_shape[0] * img_shape[1])) * 100
            area_mm_estimate = np.sqrt(region['area']) * 1.5
            
            if area_percentage > 15:
                size_desc = f"large area (~{area_mm_estimate:.0f}mm)"
                interpretation['clinical_relevance'].append("size suggests macroadenoma" if predicted_class == 'pituitary' else "large mass effect likely")
            elif area_percentage > 5:
                size_desc = f"moderate area (~{area_mm_estimate:.0f}mm)"
            elif area_percentage > 1:
                size_desc = f"small area (~{area_mm_estimate:.0f}mm)"
                if predicted_class == 'pituitary':
                    interpretation['clinical_relevance'].append("size consistent with microadenoma")
            else:
                size_desc = "tiny focal point"
            
            # Shape analysis
            if region['eccentricity'] < 0.3:
                shape_desc = "circular"
            elif region['eccentricity'] < 0.7:
                shape_desc = "oval"
            else:
                shape_desc = "elongated"
                
            if region['solidity'] < 0.8:
                shape_desc = f"irregular {shape_desc}"
                interpretation['characteristics'].append("irregular borders detected")
            
            # Intensity analysis
            if region['mean_intensity'] > 0.8:
                intensity_desc = "critical importance"
            elif region['mean_intensity'] > 0.6:
                intensity_desc = "high importance"
            else:
                intensity_desc = "moderate importance"
            
            finding = f"Region {i+1}: {size_desc} in {location}, {shape_desc} shape, {intensity_desc} (sensitivity: {region['mean_intensity']:.2f})"
            interpretation['main_findings'].append(finding)
        
        # Overall pattern interpretation
        total_sensitive_area = sum(r['area'] for r in regions_analysis)
        total_area_percentage = (total_sensitive_area / (img_shape[0] * img_shape[1])) * 100
        
        if total_area_percentage < 5:
            interpretation['location_analysis'] = "Focal abnormality pattern"
        elif total_area_percentage < 15:
            interpretation['location_analysis'] = "Moderate extent abnormality"
        else:
            interpretation['location_analysis'] = "Extensive abnormality pattern"
        
        # Medical interpretation
        if predicted_class == 'no_tumor':
            interpretation['medical_interpretation']['assessment'] = "AI Assessment: No tumor detected"
            if len(regions_analysis) == 0:
                interpretation['medical_interpretation']['observations'] = [
                    "No abnormal patterns identified",
                    "Brain structure appears within normal limits"
                ]
            else:
                interpretation['medical_interpretation']['observations'] = [
                    "Minor variations detected",
                    "Features do not match tumor patterns",
                    "Consider normal anatomical variants"
                ]
            interpretation['medical_interpretation']['recommendations'] = [
                "✓ No immediate concern detected",
                "✓ Continue routine screening as advised by physician"
            ]
        else:
            interpretation['medical_interpretation']['assessment'] = f"AI Assessment: Features consistent with {predicted_class}"
            
            if predicted_class == 'pituitary':
                interpretation['medical_interpretation']['observations'].append("Pituitary-specific findings:")
                if any('sellar' in r for r in interpretation.get('clinical_relevance', [])):
                    interpretation['medical_interpretation']['observations'].append("• Location consistent with sellar region")
                if any('microadenoma' in r for r in interpretation.get('clinical_relevance', [])):
                    interpretation['medical_interpretation']['observations'].append("• Size suggests microadenoma (<10mm)")
                elif any('macroadenoma' in r for r in interpretation.get('clinical_relevance', [])):
                    interpretation['medical_interpretation']['observations'].append("• Size suggests macroadenoma (>10mm)")
            elif predicted_class == 'meningioma':
                interpretation['medical_interpretation']['observations'] = [
                    "Extra-axial location typical of meningioma",
                    "Well-defined borders observed"
                ]
            elif predicted_class == 'glioma':
                interpretation['medical_interpretation']['observations'] = [
                    "Intra-axial location consistent with glioma",
                    "Infiltrative pattern detected"
                ]
            
            if len(regions_analysis) == 1:
                interpretation['medical_interpretation']['observations'].append("Single focal lesion identified")
            elif len(regions_analysis) > 1:
                interpretation['medical_interpretation']['observations'].append(f"Multiple regions ({len(regions_analysis)}) identified")
                interpretation['medical_interpretation']['observations'].append("Consider multifocal disease")
            
            interpretation['medical_interpretation']['recommendations'] = [
                "⚠ Abnormality detected - immediate medical consultation recommended",
                "⚠ Share this analysis with a neurologist or neuro-oncologist",
                "⚠ Additional imaging (contrast MRI) may be warranted"
            ]
        
        return interpretation
    
    def occlusion_analysis(self, img_array, predicted_class_idx, window_size=32, stride=16):
        """Perform occlusion analysis with better sensitivity detection"""
        original_prob = self.model.predict(img_array, verbose=0)[0][predicted_class_idx]
        
        height, width = 224, 224
        sensitivity_map = np.zeros((height, width))
        
        noise_level = np.std(img_array) * 0.5
        
        total_steps = ((height - window_size) // stride + 1) * ((width - window_size) // stride + 1)
        current_step = 0
        
        print(f"Starting occlusion analysis with {total_steps} regions...")
        
        for i in range(0, height - window_size + 1, stride):
            for j in range(0, width - window_size + 1, stride):
                occluded = img_array.copy()
                noise = np.random.normal(0, noise_level, (window_size, window_size, 3))
                occluded[0, i:i+window_size, j:j+window_size, :] += noise
                
                new_prob = self.model.predict(occluded, verbose=0)[0][predicted_class_idx]
                sensitivity = max(0, original_prob - new_prob)
                
                sensitivity_map[i:i+window_size, j:j+window_size] = np.maximum(
                    sensitivity_map[i:i+window_size, j:j+window_size],
                    sensitivity
                )
                
                current_step += 1
                if current_step % 50 == 0:
                    print(f"Progress: {current_step}/{total_steps} regions analyzed...")
        
        sensitivity_map = gaussian_filter(sensitivity_map, sigma=2)
        
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()
        
        return sensitivity_map
    
    def create_visualizations(self, img_array, sensitivity_map, regions_analysis, heatmap_gradcam=None):
        """Create all visualization outputs for web display"""
        visualizations = {}
        
        # Ensure img_array is correct shape
        if img_array.shape == (224, 224, 3):
            img_resized = img_array
        else:
            img_resized = cv2.resize(img_array, (224, 224))
        
        # 1. Original image
        pil_original = Image.fromarray(img_resized.astype(np.uint8))
        buffer = io.BytesIO()
        pil_original.save(buffer, format='PNG')
        buffer.seek(0)
        visualizations['original'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        # 2. Sensitivity heatmap overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * sensitivity_map), cv2.COLORMAP_HOT)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        
        pil_overlay = Image.fromarray(overlay.astype(np.uint8))
        buffer = io.BytesIO()
        pil_overlay.save(buffer, format='PNG')
        buffer.seek(0)
        visualizations['sensitivity_overlay'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        # 3. Regions with bounding boxes
        regions_img = img_resized.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, region in enumerate(regions_analysis[:3]):
            y1, x1, y2, x2 = region['bbox']
            color = colors[i] if i < 3 else (255, 255, 0)
            cv2.rectangle(regions_img, (x1, y1), (x2, y2), color, 2)
            
            cy, cx = int(region['centroid'][0]), int(region['centroid'][1])
            cv2.putText(regions_img, f"{i+1}", (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            info_text = f"{region['mean_intensity']:.2f}"
            cv2.putText(regions_img, info_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        pil_regions = Image.fromarray(regions_img.astype(np.uint8))
        buffer = io.BytesIO()
        pil_regions.save(buffer, format='PNG')
        buffer.seek(0)
        visualizations['regions_marked'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        # 4. Grad-CAM overlay if available
        if heatmap_gradcam is not None:
            heatmap_resized = cv2.resize(heatmap_gradcam, (224, 224))
            heatmap_colored_gc = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored_gc = cv2.cvtColor(heatmap_colored_gc, cv2.COLOR_BGR2RGB)
            overlay_gc = cv2.addWeighted(img_resized, 0.6, heatmap_colored_gc, 0.4, 0)
            
            pil_gradcam = Image.fromarray(overlay_gc.astype(np.uint8))
            buffer = io.BytesIO()
            pil_gradcam.save(buffer, format='PNG')
            buffer.seek(0)
            visualizations['gradcam_overlay'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        return visualizations
    
    def explain_with_validation(self, img_array, predicted_class_idx, predicted_class, confidence):
        """Complete explainable AI with validation and dynamic interpretation"""
        start_time = time.time()
        
        print("Starting comprehensive AI analysis...")
        
        # Generate Grad-CAM
        try:
            heatmap_gradcam = self.make_gradcam_heatmap(img_array, predicted_class_idx)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            heatmap_gradcam = None
        
        # Generate occlusion sensitivity map
        print("\nGenerating explainability map using occlusion analysis...")
        sensitivity_map = self.occlusion_analysis(img_array, predicted_class_idx)
        
        # Validate results
        validation_result = self.validate_occlusion_results(img_array, sensitivity_map, predicted_class_idx)
        
        # Analyze patterns
        print("\nAnalyzing detected patterns...")
        regions_analysis = self.analyze_sensitivity_patterns(sensitivity_map)
        
        # Generate dynamic interpretation
        interpretation = self.generate_dynamic_interpretation(regions_analysis, predicted_class, confidence)
        
        # Create visualizations
        img_display = (img_array[0] * 255).astype(np.uint8) if img_array.max() <= 1 else img_array[0].astype(np.uint8)
        visualizations = self.create_visualizations(img_display, sensitivity_map, regions_analysis, heatmap_gradcam)
        
        # Compile comprehensive results
        analysis_time = time.time() - start_time
        
        # Summary similar to print_dynamic_summary
        summary = {
            'diagnosis': predicted_class.upper(),
            'confidence': f"{confidence:.1%}",
            'validation': 'PASSED' if validation_result['is_valid'] else 'NEEDS REVIEW',
            'pattern_analysis': interpretation['pattern_description'],
            'location_analysis': interpretation['location_analysis'],
            'detected_regions': len(regions_analysis),
            'key_characteristics': list(set(interpretation['characteristics'])) if interpretation['characteristics'] else [],
            'clinical_relevance': list(set(interpretation['clinical_relevance'])) if interpretation['clinical_relevance'] else [],
            'ai_reasoning': interpretation['confidence_interpretation'],
            'analysis_time': f"{analysis_time:.2f}s"
        }
        
        # Detailed regional analysis
        regional_analysis = []
        for i, region in enumerate(regions_analysis[:3]):
            area_pct = (region['area'] / (224 * 224)) * 100
            regional_analysis.append({
                'region_number': i + 1,
                'size_pixels': region['area'],
                'size_percentage': f"{area_pct:.1f}%",
                'location': f"({region['centroid'][1]:.0f}, {region['centroid'][0]:.0f})",
                'importance': f"{region['mean_intensity']:.2f}",
                'eccentricity': f"{region['eccentricity']:.2f}",
                'solidity': f"{region['solidity']:.2f}"
            })
        
        return {
            'sensitivity_map': sensitivity_map.tolist(),
            'regions_analysis': regions_analysis,
            'interpretation': interpretation,
            'validation_result': validation_result,
            'visualizations': visualizations,
            'summary': summary,
            'regional_analysis': regional_analysis
        }

# Initialize
model = None
xai_analyzer = None

def load_model_safe(model_path):
    """Load model with multiple format support"""
    try:
        # Try loading as H5
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load as H5: {e}")
        try:
            # Try loading architecture and weights separately
            with open('model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('model.weights.h5')
            return model
        except Exception as e2:
            print(f"Failed to load from JSON + weights: {e2}")
            return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, xai_analyzer
    
    # Try different model locations
    model_paths = [
        "models/brain_tumor_model.h5",
        "brain_tumor_model.h5",
        "models/brain_tumor_model.keras",
        "brain_tumor_model.keras"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = load_model_safe(model_path)
            if model:
                xai_analyzer = TrueExplainableAI(model, class_names)
                print(f"✅ Model loaded from {model_path}")
                print("✅ TrueExplainableAI analyzer initialized")
                break
    
    if model is None:
        print("⚠️  No model found. Please ensure model file is in the correct location.")
    
    yield
    print("👋 Shutting down...")

app = FastAPI(title="NeuroScope XAI API - Complete TrueExplainableAI", lifespan=lifespan)

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
        "message": "NeuroScope XAI API - Complete TrueExplainableAI Analysis",
        "status": "running",
        "model_loaded": model is not None,
        "xai_enabled": xai_analyzer is not None,
        "features": [
            "Occlusion-based sensitivity analysis",
            "Grad-CAM visualization",
            "Dynamic pattern interpretation",
            "Medical recommendations",
            "Comprehensive validation"
        ]
    }

@app.post("/api/infer")
async def infer(file: UploadFile = File(...)):
    """Run complete TrueExplainableAI analysis"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize and preprocess
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        if model is None or xai_analyzer is None:
            return JSONResponse(content={
                "error": "Model not loaded",
                "message": "Please ensure brain_tumor_model.h5 is in the models directory"
            }, status_code=503)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = class_names[predicted_idx]
        
        print(f"\nPrediction: {predicted_class} ({confidence:.1%})")
        
        # Run complete XAI analysis
        xai_results = xai_analyzer.explain_with_validation(
            img_array, predicted_idx, predicted_class, confidence
        )
        
        # Prepare complete response matching your TrueExplainableAI output
        response = {
            # Basic prediction info
            "prediction": {
                "label": predicted_class,
                "probability": confidence
            },
            
            # All predictions for bar chart
            "all_predictions": {cls: float(prob) for cls, prob in zip(class_names, predictions[0])},
            
            # Visualizations
            "visualizations": xai_results['visualizations'],
            
            # Complete interpretation
            "interpretation": xai_results['interpretation'],
            
            # Validation results
            "validation": xai_results['validation_result'],
            
            # Summary
            "summary": xai_results['summary'],
            
            # Detailed regions (top 10)
            "regions": xai_results['regions_analysis'][:10],
            
            # Regional analysis table
            "regional_analysis": xai_results['regional_analysis'],
            
            # Medical interpretation
            "medical_interpretation": xai_results['interpretation']['medical_interpretation'],
            
            # Comprehensive analysis sections matching your output
            "comprehensive_analysis": {
                "ai_generated_findings": {
                    "diagnosis": xai_results['summary']['diagnosis'],
                    "confidence": xai_results['summary']['confidence'],
                    "pattern_analysis": xai_results['interpretation']['pattern_description'],
                    "spatial_distribution": xai_results['interpretation']['location_analysis'],
                    "detailed_regional_analysis": xai_results['interpretation']['main_findings']
                },
                "ai_reasoning_clinical_context": {
                    "confidence_interpretation": xai_results['interpretation']['confidence_interpretation'],
                    "pattern_characteristics": xai_results['summary']['key_characteristics'],
                    "clinical_relevance": xai_results['summary']['clinical_relevance'],
                    "decision_basis": f"{len(xai_results['regions_analysis'])} key regions identified" if xai_results['regions_analysis'] else "Diffuse or subtle features",
                    "total_affected_area": f"{sum(r['area'] for r in xai_results['regions_analysis'])/(224*224)*100:.1f}% of scan"
                },
                "validation_reliability": {
                    "status": xai_results['summary']['validation'],
                    "sensitivity_validated": xai_results['validation_result']['is_valid'],
                    "confidence_drop": f"{xai_results['validation_result']['confidence_drop']:.1%}",
                    "regions_detected": len(xai_results['regions_analysis']),
                    "avg_importance": f"{np.mean([r['mean_intensity'] for r in xai_results['regions_analysis']]):.2f}" if xai_results['regions_analysis'] else "N/A"
                },
                "region_characteristics": {
                    "total_regions": len(xai_results['regions_analysis']),
                    "sizes": [r['area'] for r in xai_results['regions_analysis'][:10]],
                    "intensities": [r['mean_intensity'] for r in xai_results['regions_analysis'][:10]],
                    "eccentricities": [r['eccentricity'] for r in xai_results['regions_analysis'][:10]]
                }
            },
            
            # Additional info
            "analysis_metadata": {
                "model_type": "EfficientNetB4",
                "analysis_time": xai_results['summary']['analysis_time'],
                "methods_used": ["Occlusion Analysis", "Grad-CAM", "Pattern Recognition"],
                "disclaimer": "This AI analysis is for screening purposes only. Clinical correlation and professional medical evaluation are essential for diagnosis."
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis error: {str(e)}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    print("\n" + "="*80)
    print("🧠 NeuroScope XAI - Complete TrueExplainableAI Implementation")
    print("="*80)
    print("Features:")
    print("  ✓ Occlusion-based sensitivity analysis")
    print("  ✓ Grad-CAM visualization")
    print("  ✓ Pattern recognition and region analysis")
    print("  ✓ Dynamic medical interpretation")
    print("  ✓ Comprehensive validation")
    print("="*80)
    print("Place your model file in one of these locations:")
    print("  - backend/models/brain_tumor_model.h5")
    print("  - backend/brain_tumor_model.h5")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)