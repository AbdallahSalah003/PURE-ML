from pydantic import BaseModel

class PredictRequest(BaseModel):
    url: str

class PredictResponse(BaseModel):
    url:         str
    label:       str    # "phishing" or "legitimate"
    probability: float  # P(phishing) — 0.0 to 1.0
    confidence:  str    # "high" / "medium" / "low"
    safe:        bool