"""
AquaSense AI - Water Quality Prediction System
Primary Flask Application Server Entrypoint
"""

import sys
import os

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app import app, logger
from backend.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  [AquaSense AI] Water Quality Prediction Server")
    print("=" * 60)
    print(f"  * Web UI:       http://{HOST}:{PORT}")
    print(f"  * Prediction:   http://{HOST}:{PORT}/predict")
    print(f"  * Analysis:     http://{HOST}:{PORT}/analysis")
    print(f"  * Health API:   http://{HOST}:{PORT}/api/health")
    print(f"  * Predict API:  http://{HOST}:{PORT}/api/predict")
    print("=" * 60 + "\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
