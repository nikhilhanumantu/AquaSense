import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "aquasense_artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"

# Server configuration
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")

# Default production model
DEFAULT_MODEL = "xgboost"

# Available trained model artifacts
AVAILABLE_MODELS = {
    "xgboost": {
        "file": "xgboost.joblib",
        "name": "XGBoost Classifier",
        "description": "Gradient Boosted Trees (Optimized Recall)",
        "is_default": True
    },
    "random_forest": {
        "file": "random_forest.joblib",
        "name": "Random Forest",
        "description": "Bagged Ensemble Classifier",
        "is_default": False
    },
    "decision_tree": {
        "file": "decision_tree.joblib",
        "name": "Decision Tree",
        "description": "Single Decision Tree Classifier",
        "is_default": False
    },
    "logistic_regression": {
        "file": "logistic_regression.joblib",
        "name": "Logistic Regression",
        "description": "Linear Baseline Classifier",
        "is_default": False
    }
}

# Feature definitions aligned with model pipeline
FEATURE_NAMES = [
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

# Standard baseline guidelines (WHO & EPA 816 standards)
WATER_STANDARDS = {
    "ph": {"min": 6.5, "max": 8.5, "ideal_min": 7.0, "ideal_max": 7.8, "unit": "pH", "name": "pH Level"},
    "Hardness": {"max": 300.0, "ideal_max": 200.0, "unit": "mg/L", "name": "Hardness"},
    "Solids": {"max": 20000.0, "ideal_max": 15000.0, "unit": "ppm", "name": "Total Solids (TDS)"},
    "Chloramines": {"max": 8.0, "ideal_max": 4.0, "unit": "ppm", "name": "Chloramines"},
    "Sulfate": {"max": 400.0, "ideal_max": 250.0, "unit": "mg/L", "name": "Sulfate"},
    "Conductivity": {"max": 600.0, "ideal_max": 400.0, "unit": "μS/cm", "name": "Conductivity"},
    "Organic_carbon": {"max": 18.0, "ideal_max": 10.0, "unit": "ppm", "name": "Organic Carbon"},
    "Trihalomethanes": {"max": 80.0, "ideal_max": 60.0, "unit": "μg/L", "name": "Trihalomethanes"},
    "Turbidity": {"max": 5.0, "ideal_max": 3.0, "unit": "NTU", "name": "Turbidity"}
}
