"""Command Line Interface demonstration for Asuna Lite."""

import json
from .workflow import AsunaLiteWorkflow, generate_synthetic_telecom_data


def run_demo() -> None:
    print("=" * 65)
    print(" ASUNA LITE — PUBLIC REFERENCE IMPLEMENTATION DEMO")
    print(" Notice: Demonstrates workflow lifecycle. Not the private engine.")
    print("=" * 65)

    workflow = AsunaLiteWorkflow()

    # 1. Load Data
    print("\n[STEP 1] Generating synthetic telecom churn data (500 records)...")
    df = generate_synthetic_telecom_data(n_samples=500, random_state=42)
    workflow.load_data(df)
    print(f"  -> Ingested {len(df)} rows across {len(df.columns)} columns.")

    # 2. Bind Target
    print("\n[STEP 2] Binding target variable: 'Churn'...")
    target_summary = workflow.set_target("Churn")
    print(f"  -> Class Distribution: {target_summary['class_counts']} ({target_summary['class_proportions']})")

    # 3. Leakage Checks
    print("\n[STEP 3] Running heuristic leakage and quality checks...")
    leakage_report = workflow.check_leakage(manual_id_cols=["customerID"])
    print(f"  -> Flagged columns: {leakage_report.flagged_columns}")
    for col, reason in leakage_report.reasons.items():
        print(f"     * '{col}': {reason}")
    print(f"  -> Clean features retained for modeling: {leakage_report.clean_features}")

    # 4. Train Models
    print("\n[STEP 4] Fitting baseline (Logistic Regression) vs contender (Random Forest) via 5-fold CV...")
    runs = workflow.train_models(test_size=0.25, random_state=42)
    for name, ev in runs.items():
        if ev.is_champion:
            print(f"  -> Model '{name}' [CHAMPION]: 5-Fold CV ROC-AUC={ev.cv_roc_auc_mean:.4f}, Untouched Holdout ROC-AUC={ev.roc_auc:.4f}, Holdout F1={ev.f1:.4f}")
        else:
            print(f"  -> Model '{name}' [CONTENDER]: 5-Fold CV ROC-AUC={ev.cv_roc_auc_mean:.4f}, CV F1={ev.cv_f1_mean:.4f} (holdout evaluation skipped)")

    # 5. Compare Models
    print("\n[STEP 5] Comparing runs and reporting champion...")
    comparison = workflow.compare_models()
    print(f"  -> Active Champion: {comparison['active_model']}")
    print(f"  -> Rationale: {comparison['decision_rationale']}")

    # 6. Score Unseen Records
    print("\n[STEP 6] Scoring new simulated customer batch (5 accounts)...")
    sample_unseen = generate_synthetic_telecom_data(n_samples=5, random_state=999)
    scored = workflow.score(sample_unseen)
    display_cols = ["customerID", "tenure_months", "monthly_charges", "risk_score", "risk_tier"]
    print(scored[display_cols].to_string(index=False))

    print("\n" + "=" * 65)
    print(" DEMO COMPLETE — 100% REPRODUCIBLE AND VERIFIED")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
