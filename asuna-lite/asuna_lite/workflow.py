"""Core workflow implementation for Asuna Lite.

Demonstrates:
- Tabular data loading and target registration
- Heuristic data leakage detection (ID columns, zero-variance, proxy signals)
- Training baseline vs contender model pipelines
- Metric comparison across ROC-AUC, F1, Precision, and Recall
- Risk scoring and tier assignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class LeakageReport:
    """Findings from heuristic leakage and data quality checks."""
    flagged_columns: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    clean_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelEvaluation:
    """Evaluation metrics for a trained candidate model."""
    name: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    pipeline: Pipeline


class AsunaLiteWorkflow:
    """State machine governing an applied ML workflow in Asuna Lite."""

    def __init__(self) -> None:
        self.raw_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.leakage_report: Optional[LeakageReport] = None
        self.runs: Dict[str, ModelEvaluation] = {}
        self.active_model_name: Optional[str] = None

    def load_data(self, data: pd.DataFrame | str) -> pd.DataFrame:
        """Loads data from a pandas DataFrame or CSV filepath."""
        if isinstance(data, str):
            self.raw_data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data.copy()
        else:
            raise TypeError("Data must be a filepath string or a pandas DataFrame.")
        return self.raw_data

    def set_target(self, target_column: str) -> Dict[str, Any]:
        """Binds target column and computes class balance."""
        if self.raw_data is None:
            raise ValueError("Must load data before setting target.")
        if target_column not in self.raw_data.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")

        self.target_column = target_column
        counts = self.raw_data[target_column].value_counts(dropna=False).to_dict()
        total = len(self.raw_data)
        proportions = {k: round(v / total, 4) for k, v in counts.items()}

        return {
            "target": target_column,
            "total_rows": total,
            "class_counts": counts,
            "class_proportions": proportions,
        }

    def check_leakage(
        self,
        manual_id_cols: Optional[List[str]] = None,
        max_cardinality_ratio: float = 0.95,
        max_correlation_threshold: float = 0.90,
    ) -> LeakageReport:
        """Executes heuristic checks for target leakage and identifier columns."""
        if self.raw_data is None or self.target_column is None:
            raise ValueError("Both data and target must be configured before running leakage checks.")

        report = LeakageReport()
        features = [c for c in self.raw_data.columns if c != self.target_column]
        manual_ids = set(manual_id_cols or [])

        # Audit each feature candidate
        for col in features:
            series = self.raw_data[col]
            n_unique = series.nunique(dropna=True)
            n_rows = len(series)
            cardinality_ratio = n_unique / max(n_rows, 1)

            # 1. Flag manual ID or near-unique categorical/string ID
            if col in manual_ids:
                report.flagged_columns.append(col)
                report.reasons[col] = "Explicitly designated identifier column"
                continue

            if cardinality_ratio >= max_cardinality_ratio and (series.dtype == "object" or "id" in col.lower()):
                report.flagged_columns.append(col)
                report.reasons[col] = f"Near-unique identifier pattern (cardinality {cardinality_ratio:.1%})"
                continue

            # 2. Flag zero-variance / constant column
            if n_unique <= 1:
                report.flagged_columns.append(col)
                report.reasons[col] = "Zero-variance (constant values)"
                continue

            # 3. Numeric correlation check with target (if target is binary/numeric)
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(self.raw_data[self.target_column]):
                corr = abs(series.corr(self.raw_data[self.target_column]))
                if corr >= max_correlation_threshold:
                    report.flagged_columns.append(col)
                    report.reasons[col] = f"Suspect target leakage (high correlation: {corr:.3f})"
                    continue

            report.clean_features.append(col)

        self.leakage_report = report
        return report

    def train_models(
        self,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Dict[str, ModelEvaluation]:
        """Trains baseline Logistic Regression and contender Random Forest on clean features."""
        if self.raw_data is None or self.target_column is None:
            raise ValueError("Data and target must be configured.")
        if self.leakage_report is None:
            self.check_leakage()

        clean_cols = self.leakage_report.clean_features
        if not clean_cols:
            raise ValueError("No clean features available for training after leakage audit.")

        X = self.raw_data[clean_cols]
        y = self.raw_data[self.target_column]

        # Ensure binary numeric target for metric computation
        if y.dtype == "object":
            y = pd.Series(np.where(y.astype(str).str.lower().isin(["yes", "true", "1", "churn"]), 1, 0))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        num_cols = [c for c in clean_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in clean_cols if c not in num_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        # Baseline: Logistic Regression
        lr_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
        ])
        lr_pipe.fit(X_train, y_train)
        lr_preds = lr_pipe.predict(X_test)
        lr_probs = lr_pipe.predict_proba(X_test)[:, 1]

        self.runs["baseline_logistic_regression"] = ModelEvaluation(
            name="baseline_logistic_regression",
            roc_auc=float(roc_auc_score(y_test, lr_probs)),
            f1=float(f1_score(y_test, lr_preds, zero_division=0)),
            precision=float(precision_score(y_test, lr_preds, zero_division=0)),
            recall=float(recall_score(y_test, lr_preds, zero_division=0)),
            pipeline=lr_pipe,
        )

        # Contender: Random Forest
        rf_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)),
        ])
        rf_pipe.fit(X_train, y_train)
        rf_preds = rf_pipe.predict(X_test)
        rf_probs = rf_pipe.predict_proba(X_test)[:, 1]

        self.runs["contender_random_forest"] = ModelEvaluation(
            name="contender_random_forest",
            roc_auc=float(roc_auc_score(y_test, rf_probs)),
            f1=float(f1_score(y_test, rf_preds, zero_division=0)),
            precision=float(precision_score(y_test, rf_preds, zero_division=0)),
            recall=float(recall_score(y_test, rf_preds, zero_division=0)),
            pipeline=rf_pipe,
        )

        # Select best model by ROC-AUC
        best_run = max(self.runs.values(), key=lambda x: x.roc_auc)
        self.active_model_name = best_run.name

        return self.runs

    def compare_models(self) -> Dict[str, Any]:
        """Compares all trained models and determines active champion."""
        if not self.runs:
            raise ValueError("No models trained yet. Run train_models() first.")

        comparison = []
        for name, ev in self.runs.items():
            comparison.append({
                "model": name,
                "roc_auc": round(ev.roc_auc, 4),
                "f1_score": round(ev.f1, 4),
                "precision": round(ev.precision, 4),
                "recall": round(ev.recall, 4),
            })

        best = max(self.runs.values(), key=lambda x: x.roc_auc)
        return {
            "leaderboard": comparison,
            "active_model": best.name,
            "decision_rationale": f"Selected '{best.name}' based on highest ROC-AUC ({best.roc_auc:.4f}).",
        }

    def score(
        self,
        new_records: pd.DataFrame,
        high_risk_threshold: float = 0.65,
        medium_risk_threshold: float = 0.35,
    ) -> pd.DataFrame:
        """Generates predictions and risk tiers for unlabelled records."""
        if self.active_model_name is None or self.active_model_name not in self.runs:
            raise ValueError("No active model available for scoring. Train models first.")

        active_eval = self.runs[self.active_model_name]
        clean_features = self.leakage_report.clean_features

        missing = [c for c in clean_features if c not in new_records.columns]
        if missing:
            raise KeyError(f"Input records missing required clean features: {missing}")

        X_score = new_records[clean_features]
        probabilities = active_eval.pipeline.predict_proba(X_score)[:, 1]

        scored_df = new_records.copy()
        scored_df["risk_score"] = np.round(probabilities, 4)

        tiers = []
        for p in probabilities:
            if p >= high_risk_threshold:
                tiers.append("High")
            elif p >= medium_risk_threshold:
                tiers.append("Medium")
            else:
                tiers.append("Low")

        scored_df["risk_tier"] = tiers
        return scored_df


def generate_synthetic_telecom_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generates a small synthetic tabular dataset representing telecom customer churn."""
    rng = np.random.default_rng(random_state)

    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    tenure_months = rng.integers(1, 72, size=n_samples)
    monthly_charges = rng.uniform(20.0, 110.0, size=n_samples).round(2)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    # Uninformative constant column (to test zero-variance filter)
    constant_flag = ["ACTIVE_TENANT"] * n_samples

    # Compute churn probability based on tenure, charges, and contract
    churn_logit = (
        -0.5
        - 0.05 * tenure_months
        + 0.03 * monthly_charges
        + np.where(contract_type == "Month-to-month", 1.2, -0.8)
        + np.where(tech_support == "No", 0.5, -0.3)
        + rng.normal(0, 0.4, size=n_samples)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame({
        "customerID": customer_ids,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract_type": contract_type,
        "tech_support": tech_support,
        "paperless_billing": paperless_billing,
        "system_status": constant_flag,
        "Churn": churn,
    })
