import json
from src import train as training

def test_training_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(training, "METADATA_PATH", tmp_path / "metadata.json")
    metadata = training.train()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["random_state"] == 42
    assert 0.0 <= metadata["metrics"]["accuracy"] <= 1.0
