import json
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")

app = FastAPI(
    title="MLOps Prediction Platform",
    description="Educational inference API; not a medical diagnostic service.",
    version="1.0.0",
)

_model = None
_metadata = None

def load_artifacts():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise RuntimeError("Model artifacts are missing. Run python -m src.train first.")
        _model = joblib.load(MODEL_PATH)
        _metadata = json.loads(METADATA_PATH.read_text())

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)

@app.get("/healthz")
def health():
    return {"status": "healthy"}

@app.get("/model")
def model_info():
    try:
        load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _metadata

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    prediction = int(_model.predict([payload.features])[0])
    probabilities = _model.predict_proba([payload.features])[0].tolist()
    return {
        "prediction": prediction,
        "class_name": _metadata["target_names"][prediction],
        "probabilities": probabilities,
        "disclaimer": "Educational model output; not medical advice or diagnosis.",
    }
