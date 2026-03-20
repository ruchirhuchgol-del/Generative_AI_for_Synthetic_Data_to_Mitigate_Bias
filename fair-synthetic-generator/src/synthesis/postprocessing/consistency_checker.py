"""
Consistency Checker
===================

Checks and enforces consistency constraints in synthetic data.
Ensures logical, domain-specific, and cross-modal constraints are satisfied.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings


class BaseConstraint(ABC):
    """Abstract base class for data constraints."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get constraint name."""
        pass
    
    @abstractmethod
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if constraint is satisfied.
        
        Args:
            data: Data to check
            
        Returns:
            Tuple of (is_satisfied, details)
        """
        pass
    
    @abstractmethod
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """
        Fix constraint violations.
        
        Args:
            data: Data to fix
            
        Returns:
            Tuple of (fixed_data, fix_details)
        """
        pass


class RangeConstraint(BaseConstraint):
    """
    Constraint for value ranges.
    
    Ensures values stay within specified bounds.
    """
    
    def __init__(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_nan: bool = True
    ):
        """
        Initialize range constraint.
        
        Args:
            column: Column name to constrain
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether NaN values are allowed
        """
        self.column = column
        self.min_val = min_val if min_val is not None else -np.inf
        self.max_val = max_val if max_val is not None else np.inf
        self.allow_nan = allow_nan
    
    @property
    def name(self) -> str:
        return f"range_{self.column}"
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check range constraint."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if self.column not in data.columns:
            return True, {"error": f"Column {self.column} not found"}
        
        col_data = data[self.column]
        
        # Check for NaN
        nan_count = col_data.isna().sum()
        
        # Check range
        in_range = (col_data >= self.min_val) & (col_data <= self.max_val)
        violations = (~in_range) & (~col_data.isna())
        violation_count = violations.sum()
        
        is_satisfied = (violation_count == 0) and (self.allow_nan or nan_count == 0)
        
        return is_satisfied, {
            "violation_count": int(violation_count),
            "nan_count": int(nan_count),
            "min_value": float(col_data.min()),
            "max_value": float(col_data.max()),
            "expected_min": self.min_val,
            "expected_max": self.max_val,
        }
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """Fix by clamping values."""
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if self.column not in df.columns:
            return data, {"fixed": 0, "method": "none"}
        
        # Clamp values
        original = df[self.column].copy()
        df[self.column] = df[self.column].clip(self.min_val, self.max_val)
        
        fixed = (original != df[self.column]).sum()
        
        if is_numpy:
            return df.values, {"fixed": int(fixed), "method": "clamp"}
        return df, {"fixed": int(fixed), "method": "clamp"}


class DependencyConstraint(BaseConstraint):
    """
    Constraint for column dependencies.
    
    Ensures relationships between columns are maintained.
    """
    
    OPERATORS = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    
    def __init__(
        self,
        column_a: str,
        column_b: str,
        operator: str,
        tolerance: float = 1e-6
    ):
        """
        Initialize dependency constraint.
        
        Args:
            column_a: First column name
            column_b: Second column name
            operator: Comparison operator
            tolerance: Numerical tolerance for comparisons
        """
        self.column_a = column_a
        self.column_b = column_b
        self.operator = operator
        self.tolerance = tolerance
        
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")
    
    @property
    def name(self) -> str:
        return f"dependency_{self.column_a}_{self.operator}_{self.column_b}"
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check dependency constraint."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if self.column_a not in data.columns or self.column_b not in data.columns:
            return True, {"error": "Column(s) not found"}
        
        col_a = data[self.column_a]
        col_b = data[self.column_b]
        
        # Check dependency
        op_func = self.OPERATORS[self.operator]
        satisfied = op_func(col_a, col_b) | (np.abs(col_a - col_b) < self.tolerance)
        
        violation_count = (~satisfied).sum()
        
        return violation_count == 0, {
            "violation_count": int(violation_count),
            "violation_rate": float(violation_count / len(data)),
        }
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """Fix dependency constraint by adjustment."""
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if self.column_a not in df.columns or self.column_b not in df.columns:
            return data, {"fixed": 0, "method": "none"}
        
        col_a = df[self.column_a]
        col_b = df[self.column_b]
        
        fixed_count = 0
        
        # Adjust based on operator
        if self.operator in [">", ">="]:
            # Ensure a > b
            violations = col_a <= col_b
            df.loc[violations, self.column_a] = col_b[violations] + self.tolerance
            fixed_count = violations.sum()
        
        elif self.operator in ["<", "<="]:
            # Ensure a < b
            violations = col_a >= col_b
            df.loc[violations, self.column_a] = col_b[violations] - self.tolerance
            fixed_count = violations.sum()
        
        elif self.operator == "==":
            # Make equal
            violations = np.abs(col_a - col_b) > self.tolerance
            df.loc[violations, self.column_a] = col_b[violations]
            fixed_count = violations.sum()
        
        if is_numpy:
            return df.values, {"fixed": int(fixed_count), "method": "adjustment"}
        return df, {"fixed": int(fixed_count), "method": "adjustment"}


class UniquenessConstraint(BaseConstraint):
    """
    Constraint for unique values.
    
    Ensures specified columns have unique values.
    """
    
    def __init__(
        self,
        columns: List[str],
        allow_null: bool = True
    ):
        """
        Initialize uniqueness constraint.
        
        Args:
            columns: Columns that should have unique combinations
            allow_null: Whether null values are allowed
        """
        self.columns = columns
        self.allow_null = allow_null
    
    @property
    def name(self) -> str:
        return f"unique_{'_'.join(self.columns)}"
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check uniqueness constraint."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        missing_cols = [c for c in self.columns if c not in data.columns]
        if missing_cols:
            return True, {"error": f"Columns not found: {missing_cols}"}
        
        # Check for duplicates
        subset = data[self.columns].dropna() if not self.allow_null else data[self.columns]
        duplicates = subset.duplicated(keep=False)
        
        duplicate_count = duplicates.sum()
        unique_count = len(data) - duplicate_count
        
        return duplicate_count == 0, {
            "duplicate_count": int(duplicate_count),
            "unique_count": int(unique_count),
            "uniqueness_rate": float(unique_count / len(data)),
        }
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """Fix by removing duplicates or adding noise."""
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Find duplicates
        duplicates = df[self.columns].duplicated(keep='first')
        duplicate_indices = duplicates[duplicates].index
        
        if len(duplicate_indices) == 0:
            return data, {"fixed": 0, "method": "none"}
        
        # Add small noise to duplicate values
        for col in self.columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                noise = np.random.normal(0, 1e-6, len(duplicate_indices))
                df.loc[duplicate_indices, col] = df.loc[duplicate_indices, col] + noise
        
        if is_numpy:
            return df.values, {"fixed": int(len(duplicate_indices)), "method": "noise"}
        return df, {"fixed": int(len(duplicate_indices)), "method": "noise"}


class CategoricalConstraint(BaseConstraint):
    """
    Constraint for categorical values.
    
    Ensures values are from allowed categories.
    """
    
    def __init__(
        self,
        column: str,
        allowed_values: List[Any],
        default_value: Optional[Any] = None
    ):
        """
        Initialize categorical constraint.
        
        Args:
            column: Column name
            allowed_values: List of allowed values
            default_value: Default value for invalid entries
        """
        self.column = column
        self.allowed_values = set(allowed_values)
        self.default_value = default_value
    
    @property
    def name(self) -> str:
        return f"categorical_{self.column}"
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check categorical constraint."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if self.column not in data.columns:
            return True, {"error": f"Column {self.column} not found"}
        
        col_data = data[self.column]
        
        # Check for invalid values
        is_valid = col_data.isin(self.allowed_values) | col_data.isna()
        invalid_count = (~is_valid).sum()
        
        return invalid_count == 0, {
            "invalid_count": int(invalid_count),
            "allowed_values": list(self.allowed_values),
            "found_values": list(col_data.dropna().unique()),
        }
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """Fix by replacing invalid values."""
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if self.column not in df.columns:
            return data, {"fixed": 0, "method": "none"}
        
        col_data = df[self.column]
        is_valid = col_data.isin(self.allowed_values) | col_data.isna()
        
        invalid_count = (~is_valid).sum()
        
        if invalid_count > 0 and self.default_value is not None:
            df.loc[~is_valid, self.column] = self.default_value
        
        if is_numpy:
            return df.values, {"fixed": int(invalid_count), "method": "replace"}
        return df, {"fixed": int(invalid_count), "method": "replace"}


class CrossModalConstraint(BaseConstraint):
    """
    Constraint for cross-modal consistency.
    
    Ensures consistency between different modalities.
    """
    
    def __init__(
        self,
        modality_a: str,
        modality_b: str,
        consistency_func: Callable[[Any, Any], bool],
        fix_func: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
    ):
        """
        Initialize cross-modal constraint.
        
        Args:
            modality_a: First modality key
            modality_b: Second modality key
            consistency_func: Function to check consistency
            fix_func: Optional function to fix inconsistency
        """
        self.modality_a = modality_a
        self.modality_b = modality_b
        self.consistency_func = consistency_func
        self.fix_func = fix_func
    
    @property
    def name(self) -> str:
        return f"crossmodal_{self.modality_a}_{self.modality_b}"
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check cross-modal consistency."""
        if not isinstance(data, dict):
            return True, {"error": "Cross-modal constraint requires dictionary data"}
        
        if self.modality_a not in data or self.modality_b not in data:
            return True, {"error": "Modality not found"}
        
        modal_a = data[self.modality_a]
        modal_b = data[self.modality_b]
        
        inconsistencies = 0
        n_samples = min(len(modal_a), len(modal_b))
        
        for i in range(n_samples):
            if not self.consistency_func(modal_a[i], modal_b[i]):
                inconsistencies += 1
        
        return inconsistencies == 0, {
            "inconsistency_count": inconsistencies,
            "consistency_rate": float((n_samples - inconsistencies) / n_samples),
        }
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]], Dict[str, Any]]:
        """Fix cross-modal inconsistencies."""
        if not isinstance(data, dict):
            return data, {"fixed": 0, "method": "none"}
        
        if self.modality_a not in data or self.modality_b not in data:
            return data, {"fixed": 0, "method": "none"}
        
        if self.fix_func is None:
            return data, {"fixed": 0, "method": "none"}
        
        modal_a = data[self.modality_a].copy()
        modal_b = data[self.modality_b].copy()
        
        fixed_count = 0
        n_samples = min(len(modal_a), len(modal_b))
        
        for i in range(n_samples):
            if not self.consistency_func(modal_a[i], modal_b[i]):
                modal_a[i], modal_b[i] = self.fix_func(modal_a[i], modal_b[i])
                fixed_count += 1
        
        data[self.modality_a] = modal_a
        data[self.modality_b] = modal_b
        
        return data, {"fixed": fixed_count, "method": "fix_func"}


class ConsistencyChecker:
    """
    Comprehensive consistency checker for synthetic data.
    
    Supports:
    - Multiple constraint types
    - Automatic constraint detection
    - Batch checking and fixing
    - Detailed violation reporting
    """
    
    def __init__(
        self,
        constraints: Optional[List[BaseConstraint]] = None,
        auto_fix: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize consistency checker.
        
        Args:
            constraints: List of constraint objects
            auto_fix: Whether to automatically fix violations
            strict_mode: If True, raise error on unfixable violations
        """
        self.constraints = constraints or []
        self.auto_fix = auto_fix
        self.strict_mode = strict_mode
        
        self._violation_history = []
    
    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Add a constraint."""
        self.constraints.append(constraint)
    
    def add_range_constraint(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> None:
        """Convenience method to add range constraint."""
        self.add_constraint(RangeConstraint(column, min_val, max_val))
    
    def add_dependency_constraint(
        self,
        column_a: str,
        column_b: str,
        operator: str
    ) -> None:
        """Convenience method to add dependency constraint."""
        self.add_constraint(DependencyConstraint(column_a, column_b, operator))
    
    def add_uniqueness_constraint(
        self,
        columns: List[str]
    ) -> None:
        """Convenience method to add uniqueness constraint."""
        self.add_constraint(UniquenessConstraint(columns))
    
    def add_categorical_constraint(
        self,
        column: str,
        allowed_values: List[Any],
        default_value: Optional[Any] = None
    ) -> None:
        """Convenience method to add categorical constraint."""
        self.add_constraint(CategoricalConstraint(column, allowed_values, default_value))
    
    def check(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check all constraints.
        
        Args:
            data: Data to check
            columns: Column names (if numpy array)
            
        Returns:
            Comprehensive check report
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        report = {
            "n_constraints": len(self.constraints),
            "n_violated": 0,
            "violations": [],
            "is_consistent": True,
            "details": {},
        }
        
        for constraint in self.constraints:
            is_satisfied, details = constraint.check(data)
            
            report["details"][constraint.name] = {
                "satisfied": is_satisfied,
                **details,
            }
            
            if not is_satisfied:
                report["n_violated"] += 1
                report["violations"].append(constraint.name)
                report["is_consistent"] = False
        
        # Store in history
        self._violation_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "report": report,
        })
        
        return report
    
    def fix(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Tuple[Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Fix all constraint violations.
        
        Args:
            data: Data to fix
            columns: Column names
            
        Returns:
            Tuple of (fixed_data, fix_report)
        """
        # Convert if needed
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        report = {
            "original_samples": len(data) if isinstance(data, pd.DataFrame) else len(list(data.values())[0]),
            "fixes_applied": [],
            "total_fixed": 0,
            "remaining_violations": [],
        }
        
        # Fix each constraint
        for constraint in self.constraints:
            fixed_data, fix_details = constraint.fix(data)
            data = fixed_data
            
            report["fixes_applied"].append({
                "constraint": constraint.name,
                **fix_details,
            })
            report["total_fixed"] += fix_details.get("fixed", 0)
        
        # Re-check after fixes
        if self.strict_mode:
            final_check = self.check(data)
            if not final_check["is_consistent"]:
                warnings.warn(
                    f"Could not fix all violations: {final_check['violations']}"
                )
            report["remaining_violations"] = final_check["violations"]
        
        # Convert back if needed
        if is_numpy:
            return data.values, report
        return data, report
    
    def check_and_fix(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """
        Check and optionally fix data.
        
        Args:
            data: Data to check/fix
            columns: Column names
            
        Returns:
            Tuple of (data, report)
        """
        check_report = self.check(data, columns)
        
        if not check_report["is_consistent"] and self.auto_fix:
            fixed_data, fix_report = self.fix(data, columns)
            return fixed_data, {
                "check": check_report,
                "fix": fix_report,
            }
        
        return data, {"check": check_report, "fix": None}
    
    def infer_constraints(
        self,
        reference_data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> List[BaseConstraint]:
        """
        Automatically infer constraints from reference data.
        
        Args:
            reference_data: Reference data to analyze
            columns: Column names
            
        Returns:
            List of inferred constraints
        """
        inferred = []
        
        if isinstance(reference_data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(reference_data.shape[1])]
            reference_data = pd.DataFrame(reference_data, columns=columns)
        
        for col in reference_data.columns:
            col_data = reference_data[col]
            
            # Skip non-numeric for range inference
            if pd.api.types.is_numeric_dtype(col_data):
                # Range constraint
                min_val = col_data.min()
                max_val = col_data.max()
                inferred.append(RangeConstraint(col, min_val, max_val))
            
            # Categorical constraint for low-cardinality
            unique_values = col_data.dropna().unique()
            if len(unique_values) <= 10:
                inferred.append(CategoricalConstraint(
                    col, 
                    list(unique_values),
                    default_value=unique_values[0] if len(unique_values) > 0 else None
                ))
        
        return inferred
    
    def get_violation_history(self) -> List[Dict[str, Any]]:
        """Get history of violations."""
        return self._violation_history
    
    def clear_history(self) -> None:
        """Clear violation history."""
        self._violation_history = []
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of constraints."""
        return {
            "n_constraints": len(self.constraints),
            "constraint_types": {
                constraint.name.split("_")[0]: sum(
                    1 for c in self.constraints 
                    if c.name.startswith(constraint.name.split("_")[0])
                )
                for constraint in self.constraints
            },
            "auto_fix": self.auto_fix,
            "strict_mode": self.strict_mode,
        }
