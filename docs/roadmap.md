# Roadmap & Project Evolution

This roadmap clarifies the division between public reference artifacts and conceptual / planned capabilities.

---

## 1. Verified Public Reference Implementation (`asuna-lite/`)

- [x] Public architecture case study and interaction design specifications
- [x] Command protocol interface definitions
- [x] Mermaid system architecture and sequence flowcharts
- [x] Reproducible Scikit-learn reference implementation (`asuna-lite/`)
- [x] Automated test suite verifying pipeline reproducibility (`pytest`)
- [x] GitHub Actions CI workflow verifying multi-version Python test runs

---

## 2. Planned / Future Conceptual Design

The following capabilities are documented as architectural concepts for future iterations:

- [ ] Automated threshold optimization for custom cost-benefit matrices
- [ ] Interactive what-if counterfactual feature perturbation CLI
- [ ] Baseline population stability index (PSI) computation for drift tracking
- [ ] Direct export of model artifacts to ONNX / PMML formats

*Note: Conceptual design items are not currently implemented in public source code.*
