import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

from .config import (
    BASE_DIR,
    FRONTEND_DIR,
    HOST,
    PORT,
    DEBUG,
    DEFAULT_MODEL,
    AVAILABLE_MODELS
)
from .model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AquaSense-Backend")

# Initialize Flask application
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize ML Model Service
model_service = ModelService()

# In-memory audit history log (holds last 50 predictions)
prediction_history = []

# -------------------------------------------------------------
# Frontend Static & HTML Routes
# -------------------------------------------------------------
@app.route("/")
@app.route("/index.html")
@app.route("/main.html")
def serve_home():
    return send_from_directory(str(FRONTEND_DIR), "main.html")

@app.route("/predict")
@app.route("/predict.html")
def serve_predict():
    return send_from_directory(str(FRONTEND_DIR), "predict.html")

@app.route("/analysis")
@app.route("/analysis.html")
def serve_analysis():
    return send_from_directory(str(FRONTEND_DIR), "analysis.html")

@app.route("/result")
@app.route("/result.html")
def serve_result():
    return send_from_directory(str(FRONTEND_DIR), "result.html")

@app.route("/<path:filename>")
def serve_static(filename):
    file_path = FRONTEND_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(FRONTEND_DIR), filename)
    return jsonify({"error": f"File '{filename}' not found"}), 404

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check and service status."""
    return jsonify({
        "status": "healthy",
        "service": "AquaSense AI Prediction Backend",
        "version": "2.4.0",
        "python_version": sys.version,
        "default_model": DEFAULT_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys())
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict_water_quality():
    """Run real-time inference using trained machine learning model."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.form.to_dict()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input payload provided. Expecting JSON or form-data with water parameter values."
            }), 400

        # Model selection (optional override, defaults to XGBoost)
        model_key = data.get("model", data.get("model_key", DEFAULT_MODEL))
        if model_key not in AVAILABLE_MODELS:
            model_key = DEFAULT_MODEL

        # Execute prediction pipeline
        result = model_service.predict(data, model_key=model_key)

        # Append to audit history log
        history_entry = {
            "run_id": result["run_id"],
            "timestamp": result["timestamp"],
            "model": result["model"]["name"],
            "is_potable": result["prediction"]["is_potable"],
            "confidence": result["prediction"]["confidence"],
            "score": result["prediction"]["score"],
            "violations_count": len(result["violations"]),
            "key_params": {
                "ph": result["inputs"]["ph"],
                "solids": result["inputs"]["Solids"],
                "turbidity": result["inputs"]["Turbidity"]
            }
        }
        prediction_history.insert(0, history_entry)
        if len(prediction_history) > 50:
            prediction_history.pop()

        logger.info(f"Prediction {result['run_id']} - Potable: {result['prediction']['is_potable']} (Score: {result['prediction']['score']})")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.route("/api/models", methods=["GET"])
def get_models():
    """Retrieve benchmark metrics of all trained candidate models."""
    try:
        benchmarks = model_service.get_models_benchmark()
        return jsonify({
            "success": True,
            "active_model": DEFAULT_MODEL,
            "models": benchmarks
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/features", methods=["GET"])
def get_features():
    """Retrieve feature importance rankings from the active XGBoost model."""
    try:
        features = model_service.get_feature_importance()
        return jsonify({
            "success": True,
            "active_model": "XGBoost",
            "features": features
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/summary", methods=["GET"])
def get_dataset_summary():
    """Retrieve training dataset distribution statistics."""
    try:
        summary = model_service.get_dataset_summary()
        return jsonify({
            "success": True,
            "summary": summary
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    """Retrieve audit history of recent predictions."""
    limit = int(request.args.get("limit", 20))
    return jsonify({
        "success": True,
        "count": len(prediction_history[:limit]),
        "history": prediction_history[:limit]
    }), 200

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clear in-memory prediction audit log."""
    prediction_history.clear()
    return jsonify({"success": True, "message": "Prediction audit history cleared."}), 200

# -------------------------------------------------------------
# Global Error Handlers
# -------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return send_from_directory(str(FRONTEND_DIR), "main.html")

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error", "details": str(e)}), 500

def create_app():
    return app

if __name__ == "__main__":
    logger.info(f"Starting AquaSense Flask Server at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
