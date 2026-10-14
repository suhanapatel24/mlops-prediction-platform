# Architecture

Dataset -> deterministic split -> StandardScaler + LogisticRegression -> evaluation -> model.joblib + metadata.json -> FastAPI -> Docker -> Kubernetes

## Reproducibility
The dataset ships with scikit-learn, the split is stratified, random_state is fixed, dependencies are version-pinned, and the container build retrains the artifact.

## Training-serving consistency
The scaler and classifier are stored as one scikit-learn Pipeline, so inference uses the same preprocessing learned during training.

## Quality gate
CI retrains the model, runs unit/API tests, checks minimum accuracy and F1 thresholds, and builds the inference image. The thresholds are quality gates for this benchmark, not claims about medical usefulness.
