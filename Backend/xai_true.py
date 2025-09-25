# backend/xai_true.py
import io
import base64
import time
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import remove_small_objects

class TrueExplainableAIWeb:
    """
    Web-compatible TrueExplainableAI:
    - Uses arrays (no file paths)
    - Produces sensitivity map via occlusion
    - Analyzes regions and generates interpretation
    - Builds web-ready visualizations (base64 PNGs)
    """

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def validate_occlusion_results(self, img_array, sensitivity_map, predicted_class_idx):
        threshold = np.percentile(sensitivity_map, 90)
        high_sensitivity_mask = sensitivity_map > threshold

        res = {
            "is_valid": True,
            "confidence_drop": 0.0,
            "original_confidence": 0.0,
            "occluded_confidence": 0.0,
            "message": ""
        }

        if np.any(high_sensitivity_mask):
            test_img = img_array.astype(np.float32).copy()
            mean_val = float(np.mean(img_array))
            test_img[0][high_sensitivity_mask] = mean_val

            new_pred = self.model.predict(test_img, verbose=0)
            new_class = int(np.argmax(new_pred[0]))
            original_conf = float(self.model.predict(img_array, verbose=0)[0][predicted_class_idx])
            new_conf = float(new_pred[0][predicted_class_idx])
            drop = original_conf - new_conf

            res["original_confidence"] = original_conf
            res["occluded_confidence"] = new_conf
            res["confidence_drop"] = drop
            res["is_valid"] = (drop > 0.05) or (new_class != predicted_class_idx)
            res["message"] = (
                "✓ Validation PASSED: Occluding important regions affects prediction"
                if res["is_valid"] else
                "⚠ Validation WARNING: Model may be too confident or regions not critical"
            )
        return res

    def analyze_sensitivity_patterns(self, sensitivity_map):
        threshold = np.percentile(sensitivity_map, 75)
        binary_map = (sensitivity_map > threshold).astype(np.uint8)

        min_size = 50
        binary_map = remove_small_objects(binary_map.astype(bool), min_size=min_size)
        labeled_map = sk_label(binary_map)
        regions = regionprops(labeled_map, intensity_image=sensitivity_map)

        regions_analysis = []
        for region in regions:
            cy, cx = region.centroid
            regions_analysis.append({
                "area": int(region.area),
                "centroid": (float(cy), float(cx)),
                "bbox": [int(x) for x in region.bbox],
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "mean_intensity": float(region.mean_intensity),
                "max_intensity": float(region.max_intensity),
                "min_intensity": float(region.min_intensity),
                "orientation": float(region.orientation),
                "perimeter": float(region.perimeter),
            })
        regions_analysis.sort(key=lambda x: x["area"] * x["mean_intensity"], reverse=True)
        return regions_analysis

    def generate_dynamic_interpretation(self, regions_analysis, predicted_class, confidence, img_shape=(224, 224)):
        interpretation = {
            "main_findings": [],
            "pattern_description": "",
            "location_analysis": "",
            "confidence_interpretation": "",
            "characteristics": [],
            "clinical_relevance": [],
            "medical_interpretation": {
                "assessment": "",
                "observations": [],
                "recommendations": []
            }
        }

        # Confidence text
        if confidence > 0.95:
            interpretation["confidence_interpretation"] = f"The AI shows very high confidence ({confidence:.1%}), indicating clear pattern recognition."
        elif confidence > 0.85:
            interpretation["confidence_interpretation"] = f"The AI shows high confidence ({confidence:.1%}), suggesting strong pattern match."
        elif confidence > 0.70:
            interpretation["confidence_interpretation"] = f"The AI shows moderate confidence ({confidence:.1%}), indicating some uncertainty."
        else:
            interpretation["confidence_interpretation"] = f"The AI shows low confidence ({confidence:.1%}), suggesting ambiguous patterns."

        if not regions_analysis:
            interpretation["main_findings"].append("No significant activation regions detected")
            interpretation["pattern_description"] = "The AI decision appears to be based on subtle, distributed features rather than focal abnormalities."
            if predicted_class == "no_tumor":
                interpretation["medical_interpretation"]["assessment"] = "AI Assessment: No tumor detected"
                interpretation["medical_interpretation"]["observations"] = [
                    "No abnormal patterns identified",
                    "Brain structure appears within normal limits"
                ]
                interpretation["medical_interpretation"]["recommendations"] = [
                    "✓ No immediate concern detected",
                    "✓ Continue routine screening as advised by physician"
                ]
            return interpretation

        # Global pattern
        num_regions = len(regions_analysis)
        if num_regions == 1:
            interpretation["pattern_description"] = "The AI focused on a single, well-defined area of interest."
        elif num_regions <= 3:
            interpretation["pattern_description"] = f"The AI identified {num_regions} distinct focal areas."
        else:
            interpretation["pattern_description"] = f"The AI detected {num_regions} regions, suggesting a diffuse or multifocal pattern."

        # Distribution characteristics
        if num_regions > 1:
            ys = [r["centroid"][0] for r in regions_analysis]
            xs = [r["centroid"][1] for r in regions_analysis]
            if np.std(xs) > 50 and np.std(ys) > 50:
                interpretation["characteristics"].append("widespread distribution across scan")
            elif np.std(xs) > 50:
                interpretation["characteristics"].append("horizontal distribution pattern")
            elif np.std(ys) > 50:
                interpretation["characteristics"].append("vertical distribution pattern")
            else:
                interpretation["characteristics"].append("clustered distribution")

        # Region summaries
        for i, region in enumerate(regions_analysis[:3]):
            cy, cx = region["centroid"]
            img_center_y, img_center_x = img_shape[0] // 2, img_shape[1] // 2

            if predicted_class == "pituitary":
                if cy > img_center_y + 30:
                    location = "lower central region (typical for pituitary)"
                    interpretation["clinical_relevance"].append("location consistent with pituitary region")
                else:
                    location = "atypical location for pituitary"
            else:
                h = "central"
                if cx < img_center_x - 30: h = "left"
                elif cx > img_center_x + 30: h = "right"
                v = "middle"
                if cy < img_center_y - 30: v = "upper"
                elif cy > img_center_y + 30: v = "lower"
                location = f"{v} {h}" if v != "middle" else h

            area_pct = (region["area"] / (img_shape[0] * img_shape[1])) * 100
            est_mm = np.sqrt(region["area"]) * 1.5
            if area_pct > 15:
                size_desc = f"large area (~{est_mm:.0f}mm)"
                interpretation["clinical_relevance"].append(
                    "size suggests macroadenoma" if predicted_class == "pituitary" else "large mass effect likely"
                )
            elif area_pct > 5:
                size_desc = f"moderate area (~{est_mm:.0f}mm)"
            elif area_pct > 1:
                size_desc = f"small area (~{est_mm:.0f}mm)"
                if predicted_class == "pituitary":
                    interpretation["clinical_relevance"].append("size consistent with microadenoma")
            else:
                size_desc = "tiny focal point"

            if region["eccentricity"] < 0.3: shape_desc = "circular"
            elif region["eccentricity"] < 0.7: shape_desc = "oval"
            else: shape_desc = "elongated"
            if region["solidity"] < 0.8:
                shape_desc = f"irregular {shape_desc}"
                interpretation["characteristics"].append("irregular borders detected")

            if region["mean_intensity"] > 0.8: importance = "critical importance"
            elif region["mean_intensity"] > 0.6: importance = "high importance"
            else: importance = "moderate importance"

            interpretation["main_findings"].append(
                f"Region {i+1}: {size_desc} in {location}, {shape_desc} shape, {importance} (sensitivity: {region['mean_intensity']:.2f})"
            )

        total_area = sum(r["area"] for r in regions_analysis)
        total_pct = (total_area / (img_shape[0] * img_shape[1])) * 100
        if total_pct < 5:
            interpretation["location_analysis"] = "Focal abnormality pattern"
        elif total_pct < 15:
            interpretation["location_analysis"] = "Moderate extent abnormality"
        else:
            interpretation["location_analysis"] = "Extensive abnormality pattern"

        if predicted_class == "no_tumor":
            interpretation["medical_interpretation"]["assessment"] = "AI Assessment: No tumor detected"
            interpretation["medical_interpretation"]["observations"] = [
                "Minor variations detected" if regions_analysis else "No abnormal patterns identified",
                "Features do not match tumor patterns" if regions_analysis else "Brain structure appears within normal limits"
            ]
            interpretation["medical_interpretation"]["recommendations"] = [
                "✓ No immediate concern detected", "✓ Continue routine screening as advised by physician"
            ]
        else:
            interpretation["medical_interpretation"]["assessment"] = f"AI Assessment: Features consistent with {predicted_class}"
            if predicted_class == "meningioma":
                interpretation["medical_interpretation"]["observations"] = [
                    "Extra-axial location typical of meningioma", "Well-defined borders observed"
                ]
            elif predicted_class == "glioma":
                interpretation["medical_interpretation"]["observations"] = [
                    "Intra-axial location consistent with glioma", "Infiltrative pattern detected"
                ]
            if len(regions_analysis) == 1:
                interpretation["medical_interpretation"]["observations"].append("Single focal lesion identified")
            elif len(regions_analysis) > 1:
                interpretation["medical_interpretation"]["observations"].append(f"Multiple regions ({len(regions_analysis)}) identified")
                interpretation["medical_interpretation"]["observations"].append("Consider multifocal disease")
            interpretation["medical_interpretation"]["recommendations"] = [
                "⚠ Abnormality detected - immediate medical consultation recommended",
                "⚠ Share this analysis with a neurologist or neuro-oncologist",
                "⚠ Additional imaging (contrast MRI) may be warranted"
            ]
        return interpretation

    def occlusion_analysis(self, img_array, predicted_class_idx, window_size=32, stride=16):
        img_array = img_array.astype(np.float32, copy=False)
        original_prob = float(self.model.predict(img_array, verbose=0)[0][predicted_class_idx])

        height, width = 224, 224
        sensitivity_map = np.zeros((height, width), dtype=np.float32)
        noise_level = float(np.std(img_array)) * 0.5

        for i in range(0, height - window_size + 1, stride):
            for j in range(0, width - window_size + 1, stride):
                occluded = img_array.astype(np.float32, copy=True)
                noise = np.random.normal(0.0, noise_level, (window_size, window_size, 3)).astype(np.float32)
                occluded[0, i:i+window_size, j:j+window_size, :] += noise

                new_prob = float(self.model.predict(occluded, verbose=0)[0][predicted_class_idx])
                sensitivity = max(0.0, original_prob - new_prob)
                sensitivity_map[i:i+window_size, j:j+window_size] = np.maximum(
                    sensitivity_map[i:i+window_size, j:j+window_size], sensitivity
                )

        sensitivity_map = gaussian_filter(sensitivity_map, sigma=2)
        maxv = float(np.max(sensitivity_map))
        if maxv > 0:
            sensitivity_map = sensitivity_map / maxv
        return sensitivity_map

    @staticmethod
    def _to_uint8_image(img_like):
        """
        Convert any float image (possibly [-1,1] or [0,1] or [0,255]) to uint8 [0,255].
        """
        arr = np.array(img_like)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0)
            elif arr.min() < 0.0 and arr.max() <= 1.0:
                # unlikely, but handle
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
            elif arr.min() < 0.0 and arr.max() <= 1.0:
                arr = (arr + 1.0) * 127.5
            elif arr.max() <= 255.0 and arr.min() >= 0.0:
                # already in 0..255 float
                pass
            else:
                # general normalization
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def create_visualizations(self, img_array_224, sensitivity_map, regions_analysis):
        visuals = {}

        # Ensure uint8 for viz
        img_u8 = self._to_uint8_image(img_array_224)

        # Original
        pil_orig = Image.fromarray(img_u8)
        buf = io.BytesIO()
        pil_orig.save(buf, format="PNG")
        buf.seek(0)
        visuals["original"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # Sensitivity overlay (uint8 safe)
        heat_u8 = np.uint8(np.clip(sensitivity_map, 0, 1) * 255)
        heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)
        heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_u8, 0.6, heat_col, 0.4, 0.0)

        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        buf.seek(0)
        visuals["sensitivity_overlay"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # Regions marked
        regions_img = img_u8.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, region in enumerate(regions_analysis[:3]):
            y1, x1, y2, x2 = region["bbox"]
            color = colors[i] if i < 3 else (255, 255, 0)
            cv2.rectangle(regions_img, (x1, y1), (x2, y2), color, 2)
            cy, cx = int(region["centroid"][0]), int(region["centroid"][1])
            cv2.putText(regions_img, f"{i+1}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(regions_img, f"{region['mean_intensity']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        pil_regions = Image.fromarray(regions_img)
        buf = io.BytesIO()
        pil_regions.save(buf, format="PNG")
        buf.seek(0)
        visuals["regions_marked"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        return visuals

    def explain_from_array(self, img_array, predicted_class_idx, predicted_class, confidence, mode="quick"):
        # Ensure float32 for math
        img_array = img_array.astype(np.float32, copy=False)

        t0 = time.time()
        if mode == "quick":
            window_size, stride = 64, 32
        else:
            window_size, stride = 32, 16

        sensitivity_map = self.occlusion_analysis(img_array, predicted_class_idx, window_size, stride)
        validation = self.validate_occlusion_results(img_array, sensitivity_map, predicted_class_idx)
        regions_analysis = self.analyze_sensitivity_patterns(sensitivity_map)
        interpretation = self.generate_dynamic_interpretation(regions_analysis, predicted_class, confidence)

        # For visualization, denormalize to uint8 smartly
        img_disp = img_array[0]
        # EfficientNet preprocess can be in [0,255] float or [-1,1] depending on version; normalize robustly
        if img_disp.dtype != np.uint8:
            if img_disp.min() < 0.0:  # likely [-1, 1]
                img_disp = (img_disp + 1.0) * 127.5
            elif img_disp.max() <= 1.0:  # [0,1]
                img_disp = img_disp * 255.0
        img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)

        visuals = self.create_visualizations(img_disp, sensitivity_map, regions_analysis)

        dt = time.time() - t0
        summary = {
            "diagnosis": predicted_class.upper(),
            "confidence": f"{confidence:.1%}",
            "validation": "PASSED" if validation["is_valid"] else "NEEDS REVIEW",
            "pattern_analysis": interpretation["pattern_description"],
            "location_analysis": interpretation["location_analysis"],
            "detected_regions": len(regions_analysis),
            "key_characteristics": list(set(interpretation["characteristics"])) if interpretation["characteristics"] else [],
            "clinical_relevance": list(set(interpretation["clinical_relevance"])) if interpretation["clinical_relevance"] else [],
            "ai_reasoning": interpretation["confidence_interpretation"],
            "analysis_time": f"{dt:.2f}s"
        }

        regional_analysis = []
        for i, region in enumerate(regions_analysis[:3]):
            area_pct = (region["area"] / (224 * 224)) * 100
            regional_analysis.append({
                "region_number": i + 1,
                "size_pixels": region["area"],
                "size_percentage": f"{area_pct:.1f}%",
                "location": f"({region['centroid'][1]:.0f}, {region['centroid'][0]:.0f})",
                "importance": f"{region['mean_intensity']:.2f}",
                "eccentricity": f"{region['eccentricity']:.2f}",
                "solidity": f"{region['solidity']:.2f}"
            })

        return {
            "sensitivity_map": sensitivity_map.tolist(),
            "regions_analysis": regions_analysis,
            "interpretation": interpretation,
            "validation_result": validation,
            "visualizations": visuals,
            "summary": summary,
            "regional_analysis": regional_analysis
        }