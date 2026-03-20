"""
Tests Module
============

Comprehensive test suite for Fair Synthetic Data Generator.

Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for pipelines
- e2e/: End-to-end workflow tests
"""

import pytest
from pathlib import Path

# Test directories
TEST_DIR = Path(__file__).parent
UNIT_DIR = TEST_DIR / "unit"
INTEGRATION_DIR = TEST_DIR / "integration"
E2E_DIR = TEST_DIR / "e2e"
