# Demo Session

This is a simplified public example of how Asuna ML Agent is intended to be used.

```text
/project new telecom_churn_demo
/load customers.csv
/target churn
/leakage
/train
/compare
/score new_customers.csv
/whatif customer_1023 monthly_fee=-10
/roi
/prescribe
/executive
Expected Workflow
The user creates or selects a project.
A dataset is loaded.
The target variable is defined.
Leakage and data quality checks are reviewed.
Models are trained and compared.
The best run is activated.
New records are scored.
Business recommendations are generated.

This demo is illustrative only. It does not include the private engine implementation.
