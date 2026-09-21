# Technical Scope & System Boundaries

Asuna ML Agent is an applied machine learning workflow assistant intended for tabular and predictive analytics in business decision-making.

---

## What Asuna Is

1. **Structured ML Workflow Orchestrator**: Guides users through hypothesis formulation, dataset profiling, leakage prevention, training, and model comparison.
2. **Business Decisioning Assistant**: Focuses on translation of probabilities into decision thresholds, risk segments, and actionable interventions.
3. **Leakage & Quality Auditor**: Evaluates predictive pipelines against standard pitfalls (target leakage, identifier memorization, lookahead bias).

---

## What Asuna Is NOT

To maintain technical credibility and clear architectural boundaries, Asuna is explicitly **not**:

- **A General-Purpose LLM Chatbot**: It does not answer conversational trivia, generate web scrapers, or act as a creative writer.
- **A Fully Autonomous AutoML Black-Box**: It does not replace human judgment regarding feature engineering, data semantics, or domain ethics.
- **A Desktop or RPA Automation Agent**: It does not control browser windows, mouse pointers, or operating system GUIs.
- **A Cloud Infrastructure Provisioner**: It does not provision Kubernetes clusters, cloud VMs, or manage Terraform scripts.

---

## Target Users & Scenarios

- **Data Scientists & ML Engineers**: Seeking structured, reproducible pipelines and systematic checks against premature model deployment.
- **Analytics Translators & Product Managers**: Requiring clear bridges between ROC-AUC curves and financial ROI / churn prevention metrics.
