"""
Privacy Metrics
===============

Base classes and specialized checkers for privacy evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

class PrivacyMetric(ABC):
    """Base class for all privacy metrics."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, real_data: Any, synthetic_data: Any) -> Dict[str, float]:
        """Compute the privacy metric."""
        pass

class KAnonymityChecker:
    """Check k-anonymity for tabular data."""
    def __init__(self, k: int = 5):
        self.k = k

    def check(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> bool:
        """Verify if data satisfies k-anonymity."""
        if not quasi_identifiers:
            return True
        counts = data.groupby(quasi_identifiers).size()
        return (counts >= self.k).all()

class LDiversityChecker:
    """Check l-diversity for tabular data."""
    def __init__(self, l: int = 2):
        self.l = l

    def check(self, data: pd.DataFrame, quasi_identifiers: List[str], sensitive_attr: str) -> bool:
        """Verify if data satisfies l-diversity."""
        if not quasi_identifiers:
            return True
        # Group by QIs and count unique sensitive values
        unique_sensitive = data.groupby(quasi_identifiers)[sensitive_attr].nunique()
        return (unique_sensitive >= self.l).all()

class PrivacyReport:
    """Container for multiple privacy evaluation results."""
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def summary(self) -> str:
        """Return a formatted summary of the report."""
        summary_str = "Privacy Evaluation Summary\n"
        summary_str += "===========================\n"
        for key, val in self.results.items():
            summary_str += f"{key}: {val}\n"
        return summary_str
