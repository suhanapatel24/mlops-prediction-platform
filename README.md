# MLOps Prediction Platform

An end-to-end, reproducible machine-learning project that trains a classifier, records model metadata, serves predictions through FastAPI, and packages the service for cloud-native delivery.

## Use case

The project uses scikit-learn's built-in Breast Cancer Wisconsin dataset to demonstrate the MLOps lifecycle without requiring private data or credentials. The application is an educational prediction service, not a medical diagnostic system.

## Pipeline

Dataset -> validation/split -> preprocessing + model training -> evaluation -> versioned artifact + metadata -> FastAPI inference -> container -> CI -> Kubernetes

## Train

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

Training writes:
- `artifacts/model.joblib`
- `artifacts/metadata.json`

These generated artifacts are intentionally ignored by Git; CI retrains before tests/builds.

## Serve

```bash
python -m src.train
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Open `/docs` for the generated OpenAPI interface.

## MLOps principles demonstrated

- reproducible data split and random seeds
- preprocessing bundled with the estimator
- explicit evaluation metrics
- model artifact + metadata contract
- API input validation
- automated model/API tests
- immutable container packaging
- deployment manifests and health probes
- evidence-based reporting: no fabricated model or deployment metrics

See `docs/model-card.md` and `docs/evidence.md`.
