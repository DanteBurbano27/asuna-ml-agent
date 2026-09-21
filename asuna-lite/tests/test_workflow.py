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

    # Both models should achieve sensible ROC-AUC (> 0.6)
    assert runs["baseline_logistic_regression"].roc_auc > 0.60
    assert runs["contender_random_forest"].roc_auc > 0.60

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
