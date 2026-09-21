# Asuna Lite — Public Reference Implementation

> **Important Notice**: Asuna Lite is a deliberately reduced public reference implementation demonstrating selected workflow concepts. It is not the private Asuna engine.

---

## Purpose

Asuna Lite provides concrete, reproducible, and verifiable evidence of the ML workflow lifecycle conceptualized in the [Asuna ML Agent Architecture](../docs/architecture.md):

```text
Load Dataset
     ↓
Set Target & Profile Balance
     ↓
Automated Leakage & Quality Checks
     ↓
Train Baseline & Candidate Models
     ↓
Compare Validation Metrics (ROC-AUC, F1, Precision, Recall)
     ↓
Batch Scoring & Risk Segmentation
```

---

## Installation & Running

### Requirements
- Python 3.10+
- `scikit-learn`, `pandas`, `pytest`

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

- **`asuna_lite.workflow.AsunaLiteWorkflow`**: State machine managing dataset loading, target binding, heuristic leakage audits, model fitting, metric comparison, and scoring inference.
- **Leakage Review Logic**: Automatically flags identifier columns (cardinality ratio near 1.0), constant features, and suspect high correlations with the target before model fitting.
- **Model Pipeline**: Employs Scikit-learn pipelines with `StandardScaler` for numeric variables and `OneHotEncoder` for categorical variables, training a Logistic Regression baseline alongside a Random Forest classifier.
- **Decision Layer**: Categorizes scored probabilities into actionable risk tiers (`High`, `Medium`, `Low`) for business interventions.
