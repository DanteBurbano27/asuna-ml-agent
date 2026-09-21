"""Core workflow implementation for Asuna Lite.

Demonstrates:
- Statistical integrity with untouched holdout isolation
- Heuristic data leakage detection on training partition only
- Training baseline vs contender model pipelines using 5-fold cross-validation
- Fold-scoped imputation and preprocessing
- Champion model selection via training CV metrics
- Single holdout evaluation on champion model
- Risk scoring and tier assignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class LeakageReport:
    """Findings from heuristic leakage and data quality checks performed on training data."""
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
    cv_roc_auc_mean: float = 0.0
    cv_roc_auc_std: float = 0.0
    cv_f1_mean: float = 0.0
    cv_precision_mean: float = 0.0
    cv_recall_mean: float = 0.0
    is_champion: bool = False
    holdout_evaluated: bool = False


class AsunaLiteWorkflow:
    """State machine governing an applied ML workflow in Asuna Lite."""

    def __init__(self) -> None:
        self.raw_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.holdout_data: Optional[pd.DataFrame] = None
        self.test_size: float = 0.25
        self.random_state: int = 42
        self.leakage_report: Optional[LeakageReport] = None
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.runs: Dict[str, ModelEvaluation] = {}
        self.active_model_name: Optional[str] = None
        self.champion_pipeline: Optional[Pipeline] = None
        self.holdout_evaluation: Optional[ModelEvaluation] = None
        self.holdout_eval_count: int = 0

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

    def split_data(
        self,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits raw dataset into training and untouched holdout partitions prior to screening or modeling."""
        if self.raw_data is None or self.target_column is None:
            raise ValueError("Both data and target must be configured before splitting.")
        self.test_size = test_size
        self.random_state = random_state

        y_raw = self.raw_data[self.target_column]
        stratify = y_raw if y_raw.nunique() > 1 else None
        self.train_data, self.holdout_data = train_test_split(
            self.raw_data, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return self.train_data, self.holdout_data

    def check_leakage(
        self,
        manual_id_cols: Optional[List[str]] = None,
        max_cardinality_ratio: float = 0.95,
        max_correlation_threshold: float = 0.90,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> LeakageReport:
        """Executes heuristic checks for target leakage and identifier columns strictly on the training partition."""
        if self.raw_data is None or self.target_column is None:
            raise ValueError("Both data and target must be configured before running leakage checks.")

        # Ensure train/holdout partition exists so screening is strictly isolated to training data
        if self.train_data is None:
            self.split_data(test_size=test_size, random_state=random_state)

        report = LeakageReport()
        features = [c for c in self.train_data.columns if c != self.target_column]
        manual_ids = set(manual_id_cols or [])

        # Audit each feature candidate on training partition ONLY
        for col in features:
            series = self.train_data[col]
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

            # 3. Numeric correlation check with target on training partition only
            target_series = self.train_data[self.target_column]
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target_series):
                corr = abs(series.corr(target_series))
                if not np.isnan(corr) and corr >= max_correlation_threshold:
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
        n_splits: int = 5,
    ) -> Dict[str, ModelEvaluation]:
        """Trains baseline Logistic Regression and contender Random Forest via training-only CV,
        selects champion model based on CV score, fits champion on full training set, and
        evaluates champion exactly once on untouched holdout.
        """
        if self.raw_data is None or self.target_column is None:
            raise ValueError("Data and target must be configured.")
        if self.train_data is None or self.holdout_data is None:
            self.split_data(test_size=test_size, random_state=random_state)
        if self.leakage_report is None:
            self.check_leakage(test_size=test_size, random_state=random_state)

        clean_cols = self.leakage_report.clean_features
        if not clean_cols:
            raise ValueError("No clean features available for training after leakage audit.")

        # Prepare X and y for train and untouched holdout
        X_train = self.train_data[clean_cols].copy()
        y_train = self.train_data[self.target_column].copy()
        X_holdout = self.holdout_data[clean_cols].copy()
        y_holdout = self.holdout_data[self.target_column].copy()

        # Convert target to binary numeric if string/object
        if y_train.dtype == "object":
            y_train = pd.Series(np.where(y_train.astype(str).str.lower().isin(["yes", "true", "1", "churn"]), 1, 0), index=X_train.index)
        if y_holdout.dtype == "object":
            y_holdout = pd.Series(np.where(y_holdout.astype(str).str.lower().isin(["yes", "true", "1", "churn"]), 1, 0), index=X_holdout.index)

        num_cols = [c for c in clean_cols if pd.api.types.is_numeric_dtype(X_train[c])]
        cat_cols = [c for c in clean_cols if c not in num_cols]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols),
            ]
        )

        candidate_definitions = {
            "baseline_logistic_regression": Pipeline([
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
            ]),
            "contender_random_forest": Pipeline([
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)),
            ]),
        }

        # Candidate model selection using 5-fold training-only cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        candidate_scores: Dict[str, Dict[str, float]] = {}

        for model_name, pipe_template in candidate_definitions.items():
            fold_aucs: List[float] = []
            fold_f1s: List[float] = []
            fold_precisions: List[float] = []
            fold_recalls: List[float] = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                fold_pipe = clone(pipe_template)
                fold_pipe.fit(X_tr, y_tr)
                val_probs = fold_pipe.predict_proba(X_val)[:, 1]
                val_preds = fold_pipe.predict(X_val)

                fold_aucs.append(float(roc_auc_score(y_val, val_probs)))
                fold_f1s.append(float(f1_score(y_val, val_preds, zero_division=0)))
                fold_precisions.append(float(precision_score(y_val, val_preds, zero_division=0)))
                fold_recalls.append(float(recall_score(y_val, val_preds, zero_division=0)))

            candidate_scores[model_name] = {
                "cv_roc_auc_mean": float(np.mean(fold_aucs)),
                "cv_roc_auc_std": float(np.std(fold_aucs)),
                "cv_f1_mean": float(np.mean(fold_f1s)),
                "cv_precision_mean": float(np.mean(fold_precisions)),
                "cv_recall_mean": float(np.mean(fold_recalls)),
            }

        self.cv_results = candidate_scores

        # Select champion model strictly based on training CV ROC-AUC
        champion_name = max(candidate_scores, key=lambda k: candidate_scores[k]["cv_roc_auc_mean"])
        self.active_model_name = champion_name

        # Fit champion model on complete training partition
        champion_pipe = candidate_definitions[champion_name]
        champion_pipe.fit(X_train, y_train)
        self.champion_pipeline = champion_pipe

        # Evaluate champion model EXACTLY ONCE on untouched holdout
        self.holdout_eval_count += 1
        holdout_probs = champion_pipe.predict_proba(X_holdout)[:, 1]
        holdout_preds = champion_pipe.predict(X_holdout)

        self.runs = {}
        for model_name in candidate_definitions:
            scores = candidate_scores[model_name]
            if model_name == champion_name:
                eval_obj = ModelEvaluation(
                    name=champion_name,
                    roc_auc=float(roc_auc_score(y_holdout, holdout_probs)),
                    f1=float(f1_score(y_holdout, holdout_preds, zero_division=0)),
                    precision=float(precision_score(y_holdout, holdout_preds, zero_division=0)),
                    recall=float(recall_score(y_holdout, holdout_preds, zero_division=0)),
                    pipeline=champion_pipe,
                    cv_roc_auc_mean=scores["cv_roc_auc_mean"],
                    cv_roc_auc_std=scores["cv_roc_auc_std"],
                    cv_f1_mean=scores["cv_f1_mean"],
                    cv_precision_mean=scores["cv_precision_mean"],
                    cv_recall_mean=scores["cv_recall_mean"],
                    is_champion=True,
                    holdout_evaluated=True,
                )
                self.holdout_evaluation = eval_obj
                self.runs[champion_name] = eval_obj
            else:
                contender_pipe = candidate_definitions[model_name]
                contender_pipe.fit(X_train, y_train)
                eval_obj = ModelEvaluation(
                    name=model_name,
                    roc_auc=scores["cv_roc_auc_mean"],
                    f1=scores["cv_f1_mean"],
                    precision=scores["cv_precision_mean"],
                    recall=scores["cv_recall_mean"],
                    pipeline=contender_pipe,
                    cv_roc_auc_mean=scores["cv_roc_auc_mean"],
                    cv_roc_auc_std=scores["cv_roc_auc_std"],
                    cv_f1_mean=scores["cv_f1_mean"],
                    cv_precision_mean=scores["cv_precision_mean"],
                    cv_recall_mean=scores["cv_recall_mean"],
                    is_champion=False,
                    holdout_evaluated=False,
                )
                self.runs[model_name] = eval_obj

        return self.runs

    def compare_models(self) -> Dict[str, Any]:
        """Compares all trained models and reports champion performance."""
        if not self.runs:
            raise ValueError("No models trained yet. Run train_models() first.")

        comparison = []
        for name, ev in self.runs.items():
            comparison.append({
                "model": name,
                "cv_roc_auc_mean": round(ev.cv_roc_auc_mean, 4),
                "cv_roc_auc_std": round(ev.cv_roc_auc_std, 4),
                "cv_f1_mean": round(ev.cv_f1_mean, 4),
                "is_champion": ev.is_champion,
                "holdout_evaluated": ev.holdout_evaluated,
                "holdout_roc_auc": round(ev.roc_auc, 4) if ev.holdout_evaluated else None,
                "holdout_f1": round(ev.f1, 4) if ev.holdout_evaluated else None,
            })

        champion_eval = self.runs[self.active_model_name]
        return {
            "leaderboard": comparison,
            "active_model": self.active_model_name,
            "decision_rationale": (
                f"Selected '{self.active_model_name}' as champion based on highest 5-fold training CV ROC-AUC "
                f"({champion_eval.cv_roc_auc_mean:.4f}). "
                f"Champion evaluated once on untouched holdout: ROC-AUC={champion_eval.roc_auc:.4f}."
            ),
            "holdout_evaluation": {
                "model": self.active_model_name,
                "roc_auc": round(champion_eval.roc_auc, 4),
                "f1": round(champion_eval.f1, 4),
                "precision": round(champion_eval.precision, 4),
                "recall": round(champion_eval.recall, 4),
            },
        }

    def score(
        self,
        new_records: pd.DataFrame,
        high_risk_threshold: float = 0.65,
        medium_risk_threshold: float = 0.35,
    ) -> pd.DataFrame:
        """Generates predictions and risk tiers for unlabelled records using champion pipeline."""
        if self.champion_pipeline is None or self.active_model_name is None:
            raise ValueError("No active champion model available for scoring. Train models first.")

        clean_features = self.leakage_report.clean_features
        missing = [c for c in clean_features if c not in new_records.columns]
        if missing:
            raise KeyError(f"Input records missing required clean features: {missing}")

        X_score = new_records[clean_features]
        probabilities = self.champion_pipeline.predict_proba(X_score)[:, 1]

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
