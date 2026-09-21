# Roadmap & Project Evolution

This roadmap clarifies the division between public case study artifacts and private enterprise capabilities.

---

## Public Repository Scope

The public repository serves as an architecture specification, design reference, and public demonstration of applied ML engineering principles.

- [x] Initial conceptual architecture documentation
- [x] Command interface specification
- [x] End-to-end workflow sequence diagrams
- [x] `asuna-lite`: Minimal, verified public reference implementation (`scikit-learn`)
- [x] Automated test suite verifying pipeline reproducibility
- [ ] Interactive terminal demonstration recording
- [ ] Extended data drift simulation scripts

---

## Private Engine Scope

The private implementation contains proprietary internal IP, enterprise integrations, and specialized agent orchestrators:

- Continuous model retraining triggers based on live drift thresholds
- Multi-tenant model registry and persistent artifact repository
- Automated SHAP tree-explainer integrations for tree ensembles
- Custom prescriptive optimization solvers
- Enterprise authentication, role-based access control (RBAC), and audit loggers

*Note: The private engine codebase is not licensed or released in this public showcase.*
