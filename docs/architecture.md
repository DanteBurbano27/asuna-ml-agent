# Architecture Overview

Asuna ML Agent follows a modular ML workflow architecture.

## Conceptual Modules

```text
User Command
    ↓
Command Router
    ↓
Project Context
    ↓
ML Workflow Controller
    ↓
Validation / Training / Scoring / Analysis Layer
    ↓
Business-Oriented Output
Main Conceptual Components
Command Router
Project State Manager
Dataset Validation Layer
Leakage Review Layer
Training Workflow Layer
Scoring Workflow Layer
Drift Awareness Layer
What-if Analysis Layer
Prescriptive Recommendation Layer
Historical Score Review Layer
Design Principles
Project-based execution
Deterministic workflow control
Business-first ML interpretation
Separation between analysis, scoring, and recommendation
Avoidance of data leakage
Reproducible model evaluation
Clear command-driven interaction
Private Implementation

The private implementation is not included in this public repository.
