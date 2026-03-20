# Contributing to Fair Synthetic Data Generator

First off, thank you for considering contributing to the Fair Synthetic Data Generator! It's people like you that make this tool better for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs
- Use the GitHub issue tracker to report bugs.
- Describe the expected behavior and the actual behavior.
- Provide a reproducible example (minimal code snippet).
- Include your environment details (OS, Python version, PyTorch/TensorFlow version).

### Suggesting Enhancements
- Open an issue with the tag "enhancement".
- Describe the feature and why it would be useful.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`make test`).
5. Make sure your code lints (`make lint`).
6. Issue that Pull Request!

## Development Setup

```bash
# Clone and install
git clone https://github.com/fair-synth/fair-synthetic-generator.git
cd fair-synthetic-generator
pip install -e ".[dev]"

# Install pre-commit hooks
make pre-commit-install
```

## Style Guide

- We use [Black](https://github.com/psf/black) for code formatting.
- we use [isort](https://pycqa.github.io/isort/) for import sorting.
- We use [Ruff](https://github.com/charliermarsh/ruff) for linting.
- Follow PEP 8 and use type hints for all new code.

## Documentation

- We use [MkDocs](https://www.mkdocs.org/) for documentation.
- Documentation is written in Markdown and located in the `docs/` directory.
- To preview documentation locally: `make docs`.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
