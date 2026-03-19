import joblib
import numpy as np
from pathlib import Path
from api.features  import extract_features
from api.schemas   import PredictResponse

MODEL_PATH = Path("models/XGB__S1.joblib")
model      = joblib.load(MODEL_PATH)  

def predict_url(url: str) -> PredictResponse:
    # 1. Extract features from raw URL
    features = extract_features(url)              # → shape (1, 82)

    # 2. Run model
    proba = model.predict_proba(features)[0][1]   # P(phishing)
    label = "phishing" if proba >= 0.5 else "legitimate"

    # 3. Confidence band based on probability
    if proba >= 0.85 or proba <= 0.15:
        confidence = "high"
    elif proba >= 0.65 or proba <= 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictResponse(
        url         = url,
        label       = label,
        probability = round(proba, 4),
        confidence  = confidence,
        safe        = label == "legitimate",
    )