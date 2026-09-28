# AquaSense AI - Intelligent Water Potability Prediction System

AquaSense AI is a full-stack, machine learning-powered water quality screening platform designed to assess the safety and drinkability of water samples based on 9 core physicochemical parameters conforming to **WHO & EPA** drinking water quality guidelines.

---

## Key Features

- **Trained Machine Learning Benchmark**: Evaluated across 4 distinct candidate models:
  - **XGBoost Classifier (Active Production Champion)**: Selected for superior harmonic balance (highest F1-score `0.5020`, highest potable recall among trees `48.44%`, and cost-sensitive handling of class imbalance).
  - **Random Forest Classifier**: 400-tree bagging ensemble with high precision (`64.62%`).
  - **Decision Tree Classifier**: Hierarchical interpretable rule partitioner (`max_depth = 8`).
  - **Logistic Regression**: Standardized linear baseline.
- **Dual-Engine Safety Architecture**: Combines gradient-boosted statistical modeling with deterministic WHO/EPA regulatory threshold safety gates. If toxic limits are breached (e.g. Chloramines $> 8.5\text{ ppm}$, Turbidity $> 5.0\text{ NTU}$, Sulfates $> 450\text{ mg/L}$, or $\text{pH} < 6.5$), the system attenuates potable likelihood, preventing dangerous false certifications.
- **Modern Full-Stack Dashboard**:
  - `main.html`: Live interactive simulation rig, instant preset samples, and responsive circular probability gauges.
  - `predict.html`: Full 9-parameter submission with live parameter validation badges and modal diagnostics.
  - `analysis.html`: 4-Model benchmark comparison cards, confusion matrix ($N=656$), and dataset EDA.
  - `result.html`: Parameter audit, WHO/EPA compliance tables, and prescribed purification steps (RO, Carbon Filtration, pH Neutralization).
- **Offline & Edge Resilience**: Client-side inference engine in `app.js` allows seamless evaluations even when disconnected from the backend.

---

## Evaluated Model Benchmark (Holdout Test Set: $N = 656$)

| Model Architecture | Accuracy | Precision | Recall (Potable) | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **62.50%** | **52.10%** | **48.44%** | **0.5020** | **0.6437** | 🟢 **Active Model** |
| **Random Forest** | 66.77% | 64.62% | 32.81% | 0.4352 | 0.6622 | Candidate |
| **Decision Tree** | 64.48% | 59.35% | 28.52% | 0.3852 | 0.6099 | Candidate |
| **Logistic Regression** | 52.44% | 41.46% | 53.12% | 0.4658 | 0.5474 | Baseline |

---

## Physicochemical Parameters Evaluated

1. **pH Level** ($6.50 - 8.50\text{ pH}$)
2. **Hardness** ($< 300\text{ mg/L}$)
3. **Total Dissolved Solids (TDS)** ($< 20,000\text{ ppm}$)
4. **Chloramines** ($< 4.00\text{ ppm}$)
5. **Sulfate** ($< 250\text{ mg/L}$)
6. **Electrical Conductivity** ($< 400\ \mu\text{S/cm}$)
7. **Total Organic Carbon (TOC)** ($< 10.0\text{ ppm}$)
8. **Trihalomethanes (THM)** ($< 80.0\ \mu\text{g/L}$)
9. **Turbidity** ($< 5.00\text{ NTU}$)

---

## Project Structure

```
AquaSense/
├── app.py                      # Flask backend API server & static file host
├── train_and_analyze.py        # Modular ML training, EDA, and evaluation pipeline
├── WaterPrediction.ipynb       # Interactive Jupyter Notebook for modeling & EDA
├── requirements.txt            # Python package dependencies
├── run.bat                     # Windows quick launch script
├── .gitignore                  # Git ignore rules (.venv, cache, etc.)
│
├── aquasense_artifacts/        # Serialized models and evaluation data
│   ├── water_potability_model.joblib   # Active Champion model (XGBoost)
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   ├── candidate_models.json
│   ├── model_metrics.json
│   ├── analysis_summary.json
│   └── feature_importance.json
│
├── dataset/
│   └── water_potability.csv    # 3,276 water quality records
│
└── frontend/                   # Web application interface
    ├── main.html               # Home dashboard & quick simulator
    ├── predict.html            # Parameter prediction form
    ├── analysis.html           # Model telemetry & benchmark comparisons
    ├── result.html             # Detailed diagnostic report & recommendations
    ├── app.js                  # Frontend client engine & API connector
    └── styles.css              # Custom styling & animations
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/nikhilhanumantu/AquaSense.git
cd AquaSense

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
- **Windows (Double-click)**: Simply run `run.bat`.
- **Command Line**:
  ```bash
  python app.py
  ```
- Open your browser to **`http://127.0.0.1:5000`**.

### 4. Retrain Models (Optional)
To retrain and evaluate all 4 candidate models:
```bash
python train_and_analyze.py
```

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/predict` | `POST` | Computes model prediction, safety score, and treatment recommendations |
| `/api/metadata` | `GET` | Returns champion model metadata and water standards |
| `/api/models` | `GET` | Returns 4-model benchmark comparisons and feature importances |
| `/api/analysis` | `GET` | Returns dataset exploratory analysis summary |
| `/api/health` | `GET` | Server health status |

---

