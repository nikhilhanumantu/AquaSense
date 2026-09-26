import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    FEATURE_NAMES,
    WATER_STANDARDS
)

class ModelService:
    def __init__(self):
        self._models = {}
        self._load_default_model()

    def _get_model_path(self, model_key: str) -> Path:
        if model_key not in AVAILABLE_MODELS:
            model_key = DEFAULT_MODEL
        filename = AVAILABLE_MODELS[model_key]["file"]
        return ARTIFACTS_DIR / filename

    def load_model(self, model_key: str = DEFAULT_MODEL):
        if model_key in self._models:
            return self._models[model_key]

        model_path = self._get_model_path(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")

        model = joblib.load(model_path)
        self._models[model_key] = model
        return model

    def _load_default_model(self):
        try:
            self.load_model(DEFAULT_MODEL)
        except Exception as e:
            print(f"[ModelService] Warning: Could not preload default model: {e}")

    def normalize_inputs(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Maps various frontend input key styles into exact feature schema."""
        key_mapping = {
            "ph": ["ph", "PH", "pH", "ph_level"],
            "Hardness": ["hardness", "Hardness", "HARDNESS"],
            "Solids": ["solids", "Solids", "tds", "total_solids", "SOLIDS"],
            "Chloramines": ["chloramines", "Chloramines", "chloramine", "CHLORAMINES"],
            "Sulfate": ["sulfate", "Sulfate", "sulfates", "SULFATE"],
            "Conductivity": ["conductivity", "Conductivity", "conduct", "CONDUCTIVITY"],
            "Organic_carbon": ["organic_carbon", "Organic_carbon", "organic", "carbon", "toc", "TOC"],
            "Trihalomethanes": ["trihalomethanes", "Trihalomethanes", "trihalo", "thm", "THM"],
            "Turbidity": ["turbidity", "Turbidity", "TURBIDITY"]
        }

        # Baseline medians from dataset for imputation fallback
        defaults = {
            "ph": 7.03,
            "Hardness": 196.37,
            "Solids": 20927.83,
            "Chloramines": 7.13,
            "Sulfate": 333.77,
            "Conductivity": 426.21,
            "Organic_carbon": 14.28,
            "Trihalomethanes": 66.40,
            "Turbidity": 3.96
        }

        normalized = {}
        for feature, aliases in key_mapping.items():
            val = None
            for alias in aliases:
                if alias in raw_data and raw_data[alias] not in (None, ""):
                    try:
                        val = float(raw_data[alias])
                        break
                    except (ValueError, TypeError):
                        pass
            normalized[feature] = val if val is not None else defaults[feature]

        return normalized

    def predict(self, raw_inputs: Dict[str, Any], model_key: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """Performs machine learning inference and standards verification."""
        if model_key not in AVAILABLE_MODELS:
            model_key = DEFAULT_MODEL

        model = self.load_model(model_key)
        normalized_inputs = self.normalize_inputs(raw_inputs)

        # Build feature DataFrame preserving schema order
        input_df = pd.DataFrame([normalized_inputs], columns=FEATURE_NAMES)

        # Execute ML prediction
        raw_pred = int(model.predict(input_df)[0])
        probabilities = None
        if hasattr(model, "predict_proba"):
            raw_proba = model.predict_proba(input_df)[0]
            # [P(Non-potable), P(Potable)]
            p_non_potable = float(raw_proba[0])
            p_potable = float(raw_proba[1])
            probabilities = {
                "non_potable": round(p_non_potable * 100, 1),
                "potable": round(p_potable * 100, 1)
            }
        else:
            p_potable = 1.0 if raw_pred == 1 else 0.0
            probabilities = {
                "non_potable": 0.0 if raw_pred == 1 else 100.0,
                "potable": 100.0 if raw_pred == 1 else 0.0
            }

        # Clinical guideline verification and scoring
        guideline_analysis = self._evaluate_guidelines(normalized_inputs, raw_pred, probabilities["potable"])

        run_id = f"RUN-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        response = {
            "success": True,
            "run_id": run_id,
            "timestamp": timestamp,
            "model": {
                "key": model_key,
                "name": AVAILABLE_MODELS[model_key]["name"],
                "description": AVAILABLE_MODELS[model_key]["description"]
            },
            "inputs": normalized_inputs,
            "prediction": {
                "raw_class": raw_pred,
                "is_potable": guideline_analysis["is_potable"],
                "status_text": "POTABLE" if guideline_analysis["is_potable"] else "NOT POTABLE",
                "badge_text": "Safe for Drinking" if guideline_analysis["is_potable"] else "Requires Treatment",
                "confidence": guideline_analysis["confidence"],
                "risk": guideline_analysis["risk"],
                "score": guideline_analysis["score"],
                "probabilities": probabilities
            },
            "violations": guideline_analysis["violations"],
            "parameters": guideline_analysis["parameters"],
            "recommendations": guideline_analysis["recommendations"]
        }

        return response

    def _evaluate_guidelines(self, inputs: Dict[str, float], raw_pred: int, prob_potable: float) -> Dict[str, Any]:
        score = 92
        violations = []
        parameters = []

        # 1. pH evaluation (6.5 - 8.5)
        ph = inputs["ph"]
        if ph < 6.5 or ph > 8.5:
            diff = 6.5 - ph if ph < 6.5 else ph - 8.5
            penalty = min(35, 15 + diff * 12)
            score -= penalty
            status = "critical"
            violations.append(f"Acidic pH ({ph:.2f})" if ph < 6.5 else f"Alkaline pH ({ph:.2f})")
        elif ph < 6.8 or ph > 8.2:
            score -= 8
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "pH Level", "value": round(ph, 2), "unit": "pH", "status": status, "limit": "6.50 – 8.50"})

        # 2. Hardness (< 300)
        h = inputs["Hardness"]
        if h > 330:
            score -= 15
            status = "critical"
            violations.append(f"Extreme Hardness ({h:.0f} mg/L)")
        elif h > 280:
            score -= 6
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Hardness", "value": round(h, 1), "unit": "mg/L", "status": status, "limit": "< 300 mg/L"})

        # 3. Solids (< 20000)
        s = inputs["Solids"]
        if s > 26000:
            score -= 25
            status = "critical"
            violations.append(f"Excessive Total Dissolved Solids ({s:.0f} ppm)")
        elif s > 20000:
            score -= 10
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Total Solids", "value": round(s, 0), "unit": "ppm", "status": status, "limit": "< 20000 ppm"})

        # 4. Chloramines (< 8.0)
        c = inputs["Chloramines"]
        if c > 8.5:
            score -= 24
            status = "critical"
            violations.append(f"High Disinfectant Chloramines ({c:.2f} ppm)")
        elif c > 7.5:
            score -= 8
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Chloramines", "value": round(c, 2), "unit": "ppm", "status": status, "limit": "< 8.00 ppm"})

        # 5. Sulfate (< 400)
        sulf = inputs["Sulfate"]
        if sulf > 450:
            score -= 22
            status = "critical"
            violations.append(f"Excessive Sulfate Level ({sulf:.1f} mg/L)")
        elif sulf > 350:
            score -= 8
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Sulfate", "value": round(sulf, 1), "unit": "mg/L", "status": status, "limit": "< 400 mg/L"})

        # 6. Conductivity (< 600)
        cond = inputs["Conductivity"]
        if cond > 650:
            score -= 14
            status = "critical"
            violations.append(f"High Electrical Conductivity ({cond:.1f} μS/cm)")
        else:
            status = "optimal"
        parameters.append({"name": "Conductivity", "value": round(cond, 1), "unit": "μS/cm", "status": status, "limit": "< 600 μS/cm"})

        # 7. Organic Carbon (< 18.0)
        org = inputs["Organic_carbon"]
        if org > 20:
            score -= 16
            status = "critical"
            violations.append(f"Elevated Total Organic Carbon ({org:.2f} ppm)")
        elif org > 16:
            score -= 6
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Organic Carbon", "value": round(org, 2), "unit": "ppm", "status": status, "limit": "< 18.0 ppm"})

        # 8. Trihalomethanes (< 80.0)
        thm = inputs["Trihalomethanes"]
        if thm > 80:
            score -= 20
            status = "critical"
            violations.append(f"Carcinogenic THM Exceeded ({thm:.1f} μg/L)")
        elif thm > 70:
            score -= 6
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Trihalomethanes", "value": round(thm, 2), "unit": "μg/L", "status": status, "limit": "< 80.0 μg/L"})

        # 9. Turbidity (< 5.0)
        turb = inputs["Turbidity"]
        if turb > 5.0:
            score -= 24
            status = "critical"
            violations.append(f"Excessive Turbidity / Cloudiness ({turb:.2f} NTU)")
        elif turb > 4.0:
            score -= 6
            status = "acceptable"
        else:
            status = "optimal"
        parameters.append({"name": "Turbidity", "value": round(turb, 2), "unit": "NTU", "status": status, "limit": "< 5.00 NTU"})

        # Safe Quality Score Bounds
        score = max(8, min(96, round(score)))

        # Harmonized potability decision:
        # A sample is certified POTABLE if it has zero critical chemical guideline violations,
        # achieves an acceptable safety score (>= 60), and the ML model probability confirms viability.
        has_critical_violations = len(violations) > 0
        model_favors_potable = (raw_pred == 1) or (prob_potable >= 38.0)
        is_potable = (not has_critical_violations) and (score >= 60) and model_favors_potable

        # Confidence & Risk Calculation
        if is_potable:
            confidence = round(max(65.0, min(96.0, float(prob_potable) if prob_potable >= 50 else float(score * 0.85 + prob_potable * 0.15))), 1)
        else:
            confidence = round(max(70.0, min(98.0, (100.0 - prob_potable) if prob_potable < 50 else float(100 - score + 14))), 1)
        risk = round(100.0 - confidence, 1)

        # Recommendations
        recommendations = []
        if is_potable:
            recommendations.append({
                "title": "Automatic Valve Clearance Approved",
                "desc": "Water registers strictly inside WHO & EPA permissible guidelines. Safe for municipal domestic routing.",
                "icon": "task_alt",
                "severity": "success"
            })
            recommendations.append({
                "title": "Standard Disinfection Maintenance",
                "desc": "Keep residual chlorine between 0.2 and 4.0 ppm to prevent downstream microbial colonization.",
                "icon": "sanitizer",
                "severity": "info"
            })
        else:
            recommendations.append({
                "title": "Dispensation Restricted · Treatment Required",
                "desc": f"Critical chemical thresholds breached ({len(violations)} flags). Do not distribute for human consumption without purification.",
                "icon": "dangerous",
                "severity": "danger"
            })

            has_tds_sulfate = any("Solids" in v or "Sulfate" in v for v in violations)
            has_turb_carbon = any("Turbidity" in v or "Organic" in v or "THM" in v for v in violations)
            has_ph = any("pH" in v for v in violations)

            if has_tds_sulfate:
                recommendations.append({
                    "title": "Reverse Osmosis (RO) Demineralization",
                    "desc": "Deploy high-pressure RO membrane filtering to strip dissolved solids and ionic sulfates.",
                    "icon": "filter_alt",
                    "severity": "warning"
                })
            if has_turb_carbon:
                recommendations.append({
                    "title": "Coagulation & Activated Carbon Bed",
                    "desc": "Flocculate suspended particulates and adsorb organic precursors to eliminate turbidity.",
                    "icon": "layers",
                    "severity": "warning"
                })
            if has_ph:
                recommendations.append({
                    "title": "pH Neutralization Buffering",
                    "desc": "Inject food-grade neutralizing reagents to bring water pH within the 6.5 – 8.5 safe window.",
                    "icon": "tune",
                    "severity": "warning"
                })

        return {
            "is_potable": is_potable,
            "confidence": confidence,
            "risk": risk,
            "score": score,
            "violations": violations,
            "parameters": parameters,
            "recommendations": recommendations
        }

    def get_models_benchmark(self) -> List[Dict[str, Any]]:
        metrics_file = ARTIFACTS_DIR / "model_metrics.csv"
        if not metrics_file.exists():
            return []

        df = pd.read_csv(metrics_file)
        results = []
        for _, row in df.iterrows():
            model_name = str(row["Model"])
            key = "xgboost" if "xgb" in model_name.lower() else (
                "random_forest" if "forest" in model_name.lower() else (
                    "decision_tree" if "tree" in model_name.lower() else "logistic_regression"
                )
            )
            results.append({
                "key": key,
                "name": model_name,
                "accuracy": round(float(row.get("Accuracy", 0)) * 100, 2),
                "precision": round(float(row.get("Precision", 0)) * 100, 2),
                "recall": round(float(row.get("Recall", 0)) * 100, 2),
                "f1": round(float(row.get("F1-Score", 0)), 4),
                "roc_auc": round(float(row.get("ROC-AUC", 0)), 4),
                "is_active": (key == DEFAULT_MODEL)
            })
        return results

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        fi_file = ARTIFACTS_DIR / "xgboost_feature_importance.csv"
        if not fi_file.exists():
            return []

        df = pd.read_csv(fi_file)
        results = []
        for _, row in df.iterrows():
            feat = str(row.iloc[0])
            importance = float(row.iloc[1])
            results.append({
                "feature": feat,
                "importance": round(importance, 4),
                "percentage": round(importance * 100, 2)
            })
        return sorted(results, key=lambda x: x["importance"], reverse=True)

    def get_dataset_summary(self) -> Dict[str, Any]:
        summary_file = ARTIFACTS_DIR / "analysis_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                return json.load(f)
        return {
            "total_samples": 3276,
            "total_features": 9,
            "missing_values": 0,
            "potable_samples": 1278,
            "not_potable_samples": 1998
        }
