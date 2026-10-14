# Model Card

## Purpose
This model exists to demonstrate a reproducible MLOps lifecycle using a public, built-in scikit-learn dataset. It is not intended for clinical or medical decision-making.

## Data
Breast Cancer Wisconsin dataset distributed with scikit-learn. The pipeline uses a stratified 80/20 train/test split with random_state=42.

## Model
StandardScaler followed by LogisticRegression. Preprocessing and estimator are serialized together to reduce training-serving skew.

## Metrics
Accuracy, precision, recall, and F1 are calculated during every training run and written to `artifacts/metadata.json`. Exact numbers should be reported only from an executed training run, not copied into documentation in advance.

## Limitations
This is an educational model trained on a small benchmark dataset. It does not represent a clinically validated system, real-world production dataset, fairness assessment, or medical device.
