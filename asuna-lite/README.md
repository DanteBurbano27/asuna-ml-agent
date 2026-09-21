# Asuna Lite — Public Reference Implementation

> **Important Notice**: Asuna Lite is a deliberately reduced public reference implementation demonstrating selected workflow concepts. It is not the private Asuna engine.

---

## Purpose

Asuna Lite provides concrete, reproducible, and verifiable evidence of the ML workflow lifecycle conceptualized in the [Asuna ML Agent Architecture](../docs/architecture.md):

```text
Load Dataset
     ↓
Set Target & Stratified Train/Holdout Partition (test_size=0.25)
     ↓
Automated Leakage & Quality Checks (Training Partition Only)
     ↓
5-Fold Stratified Cross-Validation for Baseline & Contender Pipelines
     ↓
Champion Model Selection via Training CV Metrics (Zero Holdout Snooping)
     ↓
Fit Champion on Full Training Set & Single Untouched Holdout Evaluation
     ↓
Batch Scoring with Imputation & Risk Tier Segmentation
```

---

## Installation & Running

### Requirements
- Python 3.10+
- `numpy`, `scikit-learn`, `pandas`, `pytest`

### Setup
```bash
cd asuna-lite
pip install -r requirements.txt
```

### Run Demonstration CLI
```bash
python -m asuna_lite.cli
```

### Run Test Suite
```bash
pytest tests/ -v
```

---

## Architecture of Asuna Lite

- **`asuna_lite.workflow.AsunaLiteWorkflow`**: State machine managing dataset loading, target binding, stratified train/holdout partitioning, heuristic leakage audits strictly on training data, 5-fold cross-validation, champion model fitting, single holdout evaluation, and scoring inference.
- **Leakage Review Logic**: Automatically flags identifier columns (cardinality ratio near 1.0), constant features, and suspect high correlations with the target on the training partition only.
- **Model Pipeline**: Employs Scikit-learn pipelines with `SimpleImputer` and `StandardScaler` for numeric variables, and `SimpleImputer` and `OneHotEncoder` for categorical variables, training a Logistic Regression baseline alongside a Random Forest classifier.
- **Statistical Boundary**: Contender models are compared and selected strictly using 5-fold training CV metrics (`cv_roc_auc_mean`). The untouched holdout test split is evaluated exactly once on the champion model.
- **Decision Layer**: Categorizes scored probabilities into actionable risk tiers (`High`, `Medium`, `Low`) for business interventions.
