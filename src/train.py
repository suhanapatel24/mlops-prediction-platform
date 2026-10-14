import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
RANDOM_STATE = 42

def train():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE, stratify=data.target
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "model_type": "StandardScaler + LogisticRegression",
        "dataset": "sklearn Breast Cancer Wisconsin",
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names),
        "random_state": RANDOM_STATE,
        "test_fraction": 0.2,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    train()
