from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from src import api
from src.train import train

def test_prediction_api():
    train()
    api._model = None
    api._metadata = None
    client = TestClient(api.app)
    assert client.get("/healthz").status_code == 200
    sample = load_breast_cancer().data[0].tolist()
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert len(body["probabilities"]) == 2

def test_rejects_wrong_feature_count():
    client = TestClient(api.app)
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422
