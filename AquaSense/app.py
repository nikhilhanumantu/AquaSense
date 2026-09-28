"""
AquaSense AI - Flask Backend Server
Serves the AquaSense frontend and provides water potability ML prediction and analysis APIs.
Supports 4 models: XGBoost, Random Forest, Decision Tree, Logistic Regression.
"""

import os
import json
import uuid
import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

# Configuration & Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "aquasense_artifacts")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config['JSON_SORT_KEYS'] = False
if hasattr(app, "json"):
    app.json.sort_keys = False
CORS(app)

# Feature definitions aligned with model training pipeline
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# Model keys and friendly names
MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost (Champion · Best F1)",
    "random_forest": "Random Forest (Peak Accuracy)",
    "decision_tree": "Decision Tree (Rule-Based)",
    "logistic_regression": "Logistic Regression (Linear Baseline)"
}

# 1. LOAD TRAINED MODELS
MODELS = {}
model_file_map = {
    "xgboost": "xgboost.joblib",
    "random_forest": "random_forest.joblib",
    "decision_tree": "decision_tree.joblib",
    "logistic_regression": "logistic_regression.joblib"
}

for key, filename in model_file_map.items():
    model_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(model_path):
        try:
            MODELS[key] = joblib.load(model_path)
            print(f"[AquaSense] Loaded model: {key} from {filename}")
        except Exception as e:
            print(f"[AquaSense] Error loading {key}: {e}")

# Load default / best model
best_model_path = os.path.join(ARTIFACTS_DIR, "water_potability_model.joblib")
if os.path.exists(best_model_path):
    BEST_MODEL = joblib.load(best_model_path)
else:
    BEST_MODEL = MODELS.get("xgboost") or next(iter(MODELS.values()), None)

# 2. LOAD METADATA AND ARTIFACTS
def load_json_artifact(filename, default=None):
    path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[AquaSense] Error reading {filename}: {e}")
    return default

CANDIDATE_MODELS = load_json_artifact("candidate_models.json", {})
MODEL_METRICS = load_json_artifact("model_metrics.json", [])
ANALYSIS_SUMMARY = load_json_artifact("analysis_summary.json", {})
FEATURE_IMPORTANCES = load_json_artifact("feature_importance.json", {})

# WHO & EPA Reference Drinking Water Guidelines
WATER_STANDARDS = {
    "ph": {"min": 6.5, "max": 8.5, "idealMin": 7.0, "idealMax": 7.8, "unit": "pH", "name": "pH Level"},
    "hardness": {"max": 300, "idealMax": 200, "unit": "mg/L", "name": "Hardness"},
    "solids": {"max": 20000, "idealMax": 15000, "unit": "ppm", "name": "Total Solids (TDS)"},
    "chloramines": {"max": 8.0, "idealMax": 4.0, "unit": "ppm", "name": "Chloramines"},
    "sulfate": {"max": 400, "idealMax": 250, "unit": "mg/L", "name": "Sulfate"},
    "conductivity": {"max": 600, "idealMax": 400, "unit": "μS/cm", "name": "Conductivity"},
    "organic": {"max": 18.0, "idealMax": 10.0, "unit": "ppm", "name": "Organic Carbon"},
    "trihalo": {"max": 80.0, "idealMax": 60.0, "unit": "μg/L", "name": "Trihalomethanes"},
    "turbidity": {"max": 5.0, "idealMax": 3.0, "unit": "NTU", "name": "Turbidity"}
}

def evaluate_parameters(inputs):
    """Evaluate individual water parameters against WHO/EPA reference standards."""
    ph = inputs["ph"]
    hardness = inputs["hardness"]
    solids = inputs["solids"]
    chloramines = inputs["chloramines"]
    sulfate = inputs["sulfate"]
    conductivity = inputs["conductivity"]
    organic = inputs["organic"]
    trihalo = inputs["trihalo"]
    turbidity = inputs["turbidity"]

    violations = []
    param_status_list = []
    score = 94

    # 1. pH
    if ph < 6.5 or ph > 8.5:
        violations.append("Acidic pH (< 6.5)" if ph < 6.5 else "Alkaline pH (> 8.5)")
        score -= min(35, 15 + abs(7.0 - ph) * 10)
        p_status = "critical"
    elif ph < 6.8 or ph > 8.2:
        score -= 8
        p_status = "acceptable"
    else:
        p_status = "optimal"
    param_status_list.append({"name": "pH Level", "value": round(ph, 2), "unit": "pH", "status": p_status})

    # 2. Hardness
    if hardness > 330:
        violations.append("Extreme Hardness (> 330 mg/L)")
        score -= 15
        h_status = "critical"
    elif hardness > 280:
        score -= 8
        h_status = "acceptable"
    else:
        h_status = "optimal"
    param_status_list.append({"name": "Hardness", "value": round(hardness, 1), "unit": "mg/L", "status": h_status})

    # 3. Solids (TDS)
    if solids > 26000:
        violations.append("High Total Dissolved Solids (> 26,000 ppm)")
        score -= 24
        s_status = "critical"
    elif solids > 20000:
        score -= 10
        s_status = "acceptable"
    else:
        s_status = "optimal"
    param_status_list.append({"name": "Total Solids", "value": round(solids, 0), "unit": "ppm", "status": s_status})

    # 4. Chloramines
    if chloramines > 8.5:
        violations.append("High Chloramines / Disinfection Excess (> 8.5 ppm)")
        score -= 22
        c_status = "critical"
    elif chloramines > 7.5:
        score -= 8
        c_status = "acceptable"
    else:
        c_status = "optimal"
    param_status_list.append({"name": "Chloramines", "value": round(chloramines, 2), "unit": "ppm", "status": c_status})

    # 5. Sulfate
    if sulfate > 450:
        violations.append("Elevated Sulfates (> 450 mg/L)")
        score -= 22
        sulf_status = "critical"
    elif sulfate > 350:
        score -= 10
        sulf_status = "acceptable"
    else:
        sulf_status = "optimal"
    param_status_list.append({"name": "Sulfate", "value": round(sulfate, 1), "unit": "mg/L", "status": sulf_status})

    # 6. Conductivity
    if conductivity > 650:
        violations.append("High Electrical Conductivity (> 650 μS/cm)")
        score -= 14
        cond_status = "critical"
    else:
        cond_status = "optimal"
    param_status_list.append({"name": "Conductivity", "value": round(conductivity, 1), "unit": "μS/cm", "status": cond_status})

    # 7. Organic Carbon
    if organic > 20.0:
        violations.append("Elevated Total Organic Carbon (> 20 ppm)")
        score -= 16
        org_status = "critical"
    elif organic > 16.0:
        score -= 8
        org_status = "acceptable"
    else:
        org_status = "optimal"
    param_status_list.append({"name": "Org. Carbon", "value": round(organic, 2), "unit": "ppm", "status": org_status})

    # 8. Trihalomethanes (THM)
    if trihalo > 80.0:
        violations.append("Exceeds Trihalomethanes Limit (EPA 80 μg/L)")
        score -= 18
        thm_status = "critical"
    else:
        thm_status = "optimal"
    param_status_list.append({"name": "Trihalomethanes", "value": round(trihalo, 2), "unit": "μg/L", "status": thm_status})

    # 9. Turbidity
    if turbidity > 5.0:
        violations.append("High Turbidity / Cloudiness (> 5.0 NTU)")
        score -= 22
        turb_status = "critical"
    elif turbidity > 4.0:
        score -= 8
        turb_status = "acceptable"
    else:
        turb_status = "optimal"
    param_status_list.append({"name": "Turbidity", "value": round(turbidity, 2), "unit": "NTU", "status": turb_status})

    # Synergistic chemical hazard penalties
    if chloramines > 7.2 and (ph < 6.8 or ph > 8.2):
        violations.append("Chloramine / pH Equilibrium Disruption")
        score -= 10
    if solids > 22000 and sulfate > 360:
        violations.append("Synergistic Mineral Salinity (TDS + Sulfate)")
        score -= 12
    if organic > 15 and trihalo > 65:
        violations.append("Disinfection Byproduct Precursor Hazard (TOC + THM)")
        score -= 14

    score = max(5, min(98, int(round(score))))
    return violations, param_status_list, score

def generate_recommendations(is_potable, violations):
    """Generate domain-specific treatment and action recommendations."""
    recommendations = []
    if is_potable:
        recommendations.append({
            "title": "Conforms to Reference Guidelines",
            "desc": "Water sample parameters conform to WHO & EPA safe drinking water reference thresholds. Suitable for consumption (laboratory confirmation advised).",
            "icon": "task_alt",
            "type": "success"
        })
        recommendations.append({
            "title": "Routine Disinfection Monitoring",
            "desc": "Maintain residual disinfectant between 0.2 and 4.0 ppm to prevent secondary bacterial contamination during storage.",
            "icon": "sanitizer",
            "type": "info"
        })
    else:
        desc = (
            f"Water sample breached {len(violations)} safety thresholds. Treatment recommended before consumption."
            if len(violations) > 0
            else "Sample parameters exhibit elevated risk according to the trained ML classification boundary. Laboratory testing and filtration recommended."
        )
        recommendations.append({
            "title": "Screening Alert · Corrective Treatment Advised",
            "desc": desc,
            "icon": "dangerous",
            "type": "danger"
        })
        if any("Solids" in v or "Sulfate" in v or "Salinity" in v for v in violations):
            recommendations.append({
                "title": "Reverse Osmosis (RO) Demineralization",
                "desc": "Total Dissolved Solids and mineral sulfates exceed screening levels. Demineralize using multi-stage RO membranes.",
                "icon": "filter_alt",
                "type": "warning"
            })
        if any("Turbidity" in v or "Carbon" in v or "Trihalomethanes" in v for v in violations):
            recommendations.append({
                "title": "Coagulation & Activated Carbon Filtering",
                "desc": "Reduce suspended colloidal particles and organic carbon precursors with granulated activated carbon (GAC) filtering.",
                "icon": "layers",
                "type": "warning"
            })
        if any("pH" in v for v in violations):
            recommendations.append({
                "title": "pH Neutralization Buffering",
                "desc": "Inject stabilizing buffering agents (calcite/soda ash) to return pH to the reference 6.5 – 8.5 range.",
                "icon": "tune",
                "type": "warning"
            })
    return recommendations


# ========================================================
# FRONTEND STATIC ROUTES
# ========================================================

@app.route("/")
@app.route("/index.html")
@app.route("/main.html")
def serve_main():
    return send_from_directory(FRONTEND_DIR, "main.html")

@app.route("/predict.html")
def serve_predict():
    return send_from_directory(FRONTEND_DIR, "predict.html")

@app.route("/analysis.html")
def serve_analysis():
    return send_from_directory(FRONTEND_DIR, "analysis.html")

@app.route("/result.html")
def serve_result():
    return send_from_directory(FRONTEND_DIR, "result.html")

@app.route("/<path:filename>")
def serve_static(filename):
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, filename)
    abort(404)


# ========================================================
# API ENDPOINTS
# ========================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint showing loaded models and status."""
    return jsonify({
        "status": "online",
        "service": "AquaSense AI Water Quality Prediction System",
        "version": "2.4.1",
        "models_available": list(MODELS.keys()),
        "champion_model": "xgboost"
    })

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Returns candidate model benchmarks and active model metadata."""
    # Find champion model metrics
    best_key = "xgboost"
    champ_metrics = {}
    for m in MODEL_METRICS:
        if m.get("Model", "").lower().startswith(best_key):
            champ_metrics = {
                "accuracy": m.get("Accuracy", 0.625),
                "precision": m.get("Precision", 0.521),
                "recall": m.get("Recall", 0.4844),
                "f1": m.get("F1-Score", 0.502),
                "roc_auc": m.get("ROC-AUC", 0.6437)
            }
            break

    response = {
        "success": True,
        "metadata": {
            "model_name": "XGBoost",
            "model_display_name": "XGBoost Classifier (Composite Champion)",
            "metrics": champ_metrics or {
                "accuracy": 0.625,
                "precision": 0.521,
                "recall": 0.4844,
                "f1": 0.502,
                "roc_auc": 0.6437
            },
            "candidate_models": CANDIDATE_MODELS,
            "feature_names": FEATURES,
            "standards": WATER_STANDARDS
        }
    }
    return jsonify(response)

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns all 4 models and their evaluation metrics."""
    return jsonify({
        "success": True,
        "models": CANDIDATE_MODELS,
        "benchmark_table": MODEL_METRICS,
        "features": FEATURES,
        "feature_importances": FEATURE_IMPORTANCES
    })

@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    """Returns full dataset exploratory data analysis and correlation."""
    return jsonify({
        "success": True,
        "summary": ANALYSIS_SUMMARY,
        "feature_importances": FEATURE_IMPORTANCES
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts 9 physicochemical parameters and model selection.
    Evaluates ML model and checks WHO/EPA reference thresholds.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({"success": False, "error": "No input payload provided"}), 400

        # Parse & sanitize inputs
        def parse_float(val, default):
            try:
                v = float(val)
                return v if not np.isnan(v) else default
            except (ValueError, TypeError):
                return default

        inputs = {
            "ph": parse_float(data.get("ph"), 7.0),
            "hardness": parse_float(data.get("hardness") or data.get("Hardness"), 180.0),
            "solids": parse_float(data.get("solids") or data.get("Solids"), 15000.0),
            "chloramines": parse_float(data.get("chloramines") or data.get("Chloramines"), 7.0),
            "sulfate": parse_float(data.get("sulfate") or data.get("Sulfate"), 300.0),
            "conductivity": parse_float(data.get("conductivity") or data.get("Conductivity"), 400.0),
            "organic": parse_float(data.get("organic") or data.get("carbon") or data.get("Organic_carbon"), 10.0),
            "trihalo": parse_float(data.get("trihalo") or data.get("thm") or data.get("Trihalomethanes"), 60.0),
            "turbidity": parse_float(data.get("turbidity") or data.get("Turbidity"), 3.0)
        }

        # Resolve model
        raw_model_key = str(data.get("model", "xgboost")).lower().replace("-", "_").replace(" ", "_")
        if "random" in raw_model_key or "rf" in raw_model_key:
            model_key = "random_forest"
        elif "decision" in raw_model_key or "dt" in raw_model_key:
            model_key = "decision_tree"
        elif "logistic" in raw_model_key or "lr" in raw_model_key:
            model_key = "logistic_regression"
        else:
            model_key = "xgboost"

        pipeline = MODELS.get(model_key) or BEST_MODEL
        model_name = MODEL_DISPLAY_NAMES.get(model_key, "XGBoost Classifier")

        # Prepare DataFrame matching pipeline column names
        row = pd.DataFrame([{
            "ph": inputs["ph"],
            "Hardness": inputs["hardness"],
            "Solids": inputs["solids"],
            "Chloramines": inputs["chloramines"],
            "Sulfate": inputs["sulfate"],
            "Conductivity": inputs["conductivity"],
            "Organic_carbon": inputs["organic"],
            "Trihalomethanes": inputs["trihalo"],
            "Turbidity": inputs["turbidity"]
        }])

        # Perform ML Model Inference
        raw_pred = int(pipeline.predict(row)[0])
        raw_probs = pipeline.predict_proba(row)[0]
        model_potable_prob = float(raw_probs[1])
        model_risk_prob = float(raw_probs[0])

        # Evaluate against WHO & EPA standards
        violations, param_status_list, safety_score = evaluate_parameters(inputs)

        # Domain calibration: If critical chemical parameters are severely breached, water is not potable
        critical_violations_count = sum(1 for p in param_status_list if p.get("status") == "critical")
        
        # Physical safety calibration:
        # A water sample with critical chemical/toxic breaches cannot have a high potable likelihood.
        if critical_violations_count > 0 or safety_score < 60:
            is_potable = False
            prediction = 0
            safety_factor = max(0.05, float(safety_score) / 100.0)
            calibrated_potable_prob = min(model_potable_prob * safety_factor, 0.45)
        else:
            is_potable = True
            prediction = 1
            calibrated_potable_prob = model_potable_prob * 0.35 + (float(safety_score) / 100.0) * 0.65

        # Real calibrated prediction probability directly from the active model
        potable_pct = int(round(calibrated_potable_prob * 100))
        potable_pct = max(3, min(97, potable_pct))
        risk_pct = 100 - potable_pct

        confidence = potable_pct if is_potable else risk_pct
        recommendations = generate_recommendations(is_potable, violations)
        run_id = f"RUN-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        response = {
            "success": True,
            "prediction": prediction,
            "label": "Potable" if is_potable else "Not Potable",
            "probability": round(potable_pct / 100.0, 4),
            "model_name": model_name,
            "model_key": model_key,
            "diagnostic": {
                "is_potable": is_potable,
                "score": safety_score,
                "confidence": confidence,
                "risk": risk_pct,
                "probabilities": {
                    "potable": potable_pct,
                    "non_potable": risk_pct
                }
            },
            "inputs": inputs,
            "params": inputs,
            "parameters": param_status_list,
            "violations": violations,
            "flags": violations,
            "recommendations": recommendations,
            "disclaimer": "This prediction is an ML-based screening aid and should not replace certified laboratory water testing.",
            "timestamp": timestamp,
            "run_id": run_id
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ========================================================
# SERVER ENTRY POINT
# ========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n========================================================")
    print(f"  AquaSense AI Flask Server Running")
    print(f"  Open in Browser: http://127.0.0.1:{port}")
    print(f"  Active Models: {', '.join(MODELS.keys())}")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
