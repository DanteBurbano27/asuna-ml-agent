# Command Reference

The Asuna ML Agent workflow is organized into four sequential functional domains: Project Setup, Validation, Training/Comparison, and Business Decisioning.

---

## 1. Project Commands

| Command | Arguments | Purpose |
|---|---|---|
| `/project` | `<name>` | Create or select an ML project workspace. |
| `/load` | `<file_path>` | Ingest a dataset into the active project context. |
| `/target` | `<column_name>` | Declare the primary target variable and analyze class balance. |
| `/date` | `<column_name>` | Designate a temporal index for out-of-time validation splits. |

---

## 2. Validation Commands

| Command | Arguments | Purpose |
|---|---|---|
| `/leakage` | *(none)* | Execute automated heuristic audits for proxy features, identifiers, and high correlation anomalies. |
| `/results` | `[run_id]` | Inspect validation metrics, confusion matrices, and feature importances. |
| `/explain` | `[record_id]` | Produce local model interpretability (e.g. SHAP-style directional feature contributions). |

---

## 3. Training & Orchestration Commands

| Command | Arguments | Purpose |
|---|---|---|
| `/train` | `[--models <list>]` | Trigger training workflow across baseline and candidate architectures. |
| `/run` | `[config_file]` | Execute a fully specified training experiment with predetermined hyperparameters. |
| `/runs` | *(none)* | List all completed experiment runs with their primary validation metrics. |
| `/activate-run` | `<run_id>` | Set a specific trained model artifact as the active production candidate. |
| `/compare` | `<run_a> <run_b>` | Generate a side-by-side metric comparison and decision-boundary trade-off analysis. |

---

## 4. Business & Prescriptive Commands

| Command | Arguments | Purpose |
|---|---|---|
| `/score` | `<input_file>` | Compute predicted probabilities and risk deciles for unlabelled records. |
| `/whatif` | `<id> <feature=val>` | Simulate feature counterfactuals and observe resulting score shifts. |
| `/history` | *(none)* | Inspect scoring logs and monitor potential population drift over time. |
| `/roi` | `[cost, benefit]` | Calculate projected net business value across variable decision thresholds. |
| `/prescribe` | `[policy_name]` | Map predicted risks to actionable business interventions (e.g., retention outreach). |
| `/executive` | *(none)* | Synthesize an executive-ready brief summarizing model health, performance, and projected ROI. |
