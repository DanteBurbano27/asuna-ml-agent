"""Comprehensive test suite for Asuna Lite workflow."""

import pytest
import pandas as pd
import numpy as np
from asuna_lite.workflow import AsunaLiteWorkflow, generate_synthetic_telecom_data


@pytest.fixture
def sample_dataset():
    return generate_synthetic_telecom_data(n_samples=200, random_state=123)


def test_workflow_initialization_and_loading(sample_dataset):
    wf = AsunaLiteWorkflow()
    loaded = wf.load_data(sample_dataset)
    assert len(loaded) == 200
    assert "Churn" in loaded.columns


def test_set_target(sample_dataset):
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    summary = wf.set_target("Churn")
    assert summary["target"] == "Churn"
    assert summary["total_rows"] == 200
    assert 0 in summary["class_counts"]
    assert 1 in summary["class_counts"]


def test_leakage_detection(sample_dataset):
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    report = wf.check_leakage(manual_id_cols=["customerID"])

    # customerID should be flagged as ID
    assert "customerID" in report.flagged_columns

    # system_status is constant, should be flagged
    assert "system_status" in report.flagged_columns

    # clean features should include actual features
    assert "tenure_months" in report.clean_features
    assert "monthly_charges" in report.clean_features
    assert "contract_type" in report.clean_features


def test_model_training_and_comparison(sample_dataset):
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    wf.check_leakage(manual_id_cols=["customerID"])

    runs = wf.train_models(test_size=0.25, random_state=42)
    assert "baseline_logistic_regression" in runs
    assert "contender_random_forest" in runs

    # Both models should achieve sensible CV ROC-AUC (> 0.6)
    assert runs["baseline_logistic_regression"].cv_roc_auc_mean > 0.60
    assert runs["contender_random_forest"].cv_roc_auc_mean > 0.60

    comparison = wf.compare_models()
    assert comparison["active_model"] in ["baseline_logistic_regression", "contender_random_forest"]
    assert len(comparison["leaderboard"]) == 2


def test_scoring_unseen_records(sample_dataset):
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    wf.check_leakage(manual_id_cols=["customerID"])
    wf.train_models(test_size=0.25, random_state=42)

    unseen = generate_synthetic_telecom_data(n_samples=10, random_state=456)
    scored = wf.score(unseen)

    assert "risk_score" in scored.columns
    assert "risk_tier" in scored.columns
    assert len(scored) == 10
    assert set(scored["risk_tier"]).issubset({"High", "Medium", "Low"})
    assert scored["risk_score"].between(0.0, 1.0).all()


def test_holdout_isolation_in_screening(sample_dataset):
    """Verifies that holdout data is never touched or leaked into screening decisions."""
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    wf.split_data(test_size=0.25, random_state=42)

    # Corrupt holdout partition by making 'tenure_months' constant in holdout only
    wf.holdout_data["tenure_months"] = 999

    # Run screening
    report = wf.check_leakage(manual_id_cols=["customerID"])

    # 'tenure_months' must NOT be flagged because training partition has varying values
    assert "tenure_months" not in report.flagged_columns
    assert "tenure_months" in report.clean_features


def test_model_selection_strictly_by_training_cv(sample_dataset):
    """Verifies champion selection is strictly decided by training cross-validation score."""
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    runs = wf.train_models(test_size=0.25, random_state=42)

    cv_scores = wf.cv_results
    expected_champion = max(cv_scores, key=lambda k: cv_scores[k]["cv_roc_auc_mean"])

    assert wf.active_model_name == expected_champion
    assert runs[expected_champion].is_champion is True


def test_single_holdout_evaluation_on_champion_only(sample_dataset):
    """Verifies holdout evaluation is strictly executed once, and exclusively on champion model."""
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    runs = wf.train_models(test_size=0.25, random_state=42)

    # Exactly 1 holdout evaluation overall
    assert wf.holdout_eval_count == 1

    # Only champion is evaluated on holdout
    champion_name = wf.active_model_name
    assert runs[champion_name].holdout_evaluated is True
    assert sum(1 for m in runs.values() if m.holdout_evaluated) == 1

    # Non-champion models must NOT touch holdout
    non_champions = [name for name in runs if name != champion_name]
    for non_champ in non_champions:
        assert runs[non_champ].holdout_evaluated is False


def test_pipeline_imputer_handles_missing_values(sample_dataset):
    """Verifies fold-scoped SimpleImputer handles unseen missing values in inference."""
    wf = AsunaLiteWorkflow()
    wf.load_data(sample_dataset)
    wf.set_target("Churn")
    wf.train_models(test_size=0.25, random_state=42)

    unseen = generate_synthetic_telecom_data(n_samples=10, random_state=789)
    # Inject missing values across numeric and categorical columns
    unseen.loc[0, "tenure_months"] = np.nan
    unseen.loc[1, "monthly_charges"] = np.nan
    unseen.loc[2, "contract_type"] = np.nan

    scored = wf.score(unseen)
    assert len(scored) == 10
    assert scored["risk_score"].notna().all()
    assert set(scored["risk_tier"]).issubset({"High", "Medium", "Low"})
