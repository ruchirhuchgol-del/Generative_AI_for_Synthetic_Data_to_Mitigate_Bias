# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-20

### Added
- Core generative architectures: FairGAN, FairDiffusion, DebiasedVAE.
- Fairness module with Group, Individual, and Counterfactual constraints.
- Multi-modal support (Tabular, Text, Image).
- FastAPI-based REST API for generation and evaluation.
- Comprehensive evaluation suite (Fidelity, Fairness, Privacy).
- Documentation boilerplate.

### Fixed
- Fixed broken imports in `src/data/`.
- Fixed empty `src/models/__init__.py`.
- Fixed `conftest.py` and test file naming inconsistencies.
- Fixed documentation pathing and Makefile references.
- Corrected notebook JSON structure and kernelspecs.
