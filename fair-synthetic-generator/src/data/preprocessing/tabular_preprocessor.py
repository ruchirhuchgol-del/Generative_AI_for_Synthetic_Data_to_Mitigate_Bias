"""
Tabular Preprocessor
====================

Preprocessing utilities for tabular data including:
- Numerical feature normalization (standard, minmax, robust, log)
- Categorical feature encoding (one-hot, label, target, embedding)
- Missing value imputation
- Outlier detection and handling
- Feature engineering
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Comprehensive preprocessor for tabular data.
    
    Handles:
    - Numerical feature normalization and scaling
    - Categorical feature encoding
    - Missing value imputation
    - Outlier detection and treatment
    - Feature-specific preprocessing
    
    Attributes:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        scalers: Dictionary of fitted scalers
        encoders: Dictionary of fitted encoders
        imputers: Dictionary of fitted imputers
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        normalization: str = "standard",
        encoding: str = "one_hot",
        imputation: str = "median",
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0
    ):
        """
        Initialize the tabular preprocessor.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target_column: Name of target variable
            sensitive_attributes: List of sensitive attribute names
            normalization: Normalization method ('standard', 'minmax', 'robust', 'log')
            encoding: Encoding method ('one_hot', 'label', 'target')
            imputation: Imputation method ('mean', 'median', 'mode', 'knn')
            handle_outliers: Whether to handle outliers
            outlier_method: Outlier detection method ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.sensitive_attributes = sensitive_attributes or []
        
        self.normalization = normalization
        self.encoding = encoding
        self.imputation = imputation
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Fitted components
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        
        # Metadata
        self._feature_order: List[str] = []
        self._is_fitted = False
        self._statistics: Dict[str, Any] = {}
        
    def fit(self, data: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training DataFrame
            
        Returns:
            Fitted preprocessor
        """
        # Auto-detect features if not specified
        if not self.numerical_features and not self.categorical_features:
            self._auto_detect_features(data)
            
        # Fit imputers
        self._fit_imputers(data)
        
        # Fit scalers for numerical features
        self._fit_scalers(data)
        
        # Fit encoders for categorical features
        self._fit_encoders(data)
        
        # Compute statistics
        self._compute_statistics(data)
        
        # Store feature order
        self._feature_order = [
            col for col in data.columns 
            if col != self.target_column
        ]
        
        self._is_fitted = True
        logger.info(f"TabularPreprocessor fitted on {len(data)} samples")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        result = data.copy()
        
        # Impute missing values
        result = self._transform_impute(result)
        
        # Handle outliers
        if self.handle_outliers:
            result = self._transform_outliers(result)
            
        # Normalize numerical features
        result = self._transform_normalize(result)
        
        # Encode categorical features
        result = self._transform_encode(result)
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Training DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform to original space.
        
        Args:
            data: Transformed data
            features: Features to inverse transform (None for all)
            
        Returns:
            DataFrame in original space
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
            
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self._get_output_columns())
            
        result = data.copy()
        features = features or list(self.scalers.keys()) + list(self.encoders.keys())
        
        # Inverse transform numerical features
        for col in self.scalers:
            if col in result.columns and col in features:
                if self.normalization == "standard":
                    result[col] = self.scalers[col].inverse_transform(
                        result[[col]]
                    ).ravel()
                elif self.normalization == "minmax":
                    result[col] = self.scalers[col].inverse_transform(
                        result[[col]]
                    ).ravel()
                    
        # Inverse transform categorical features (label encoding only)
        for col in self.encoders:
            if col in result.columns and col in features:
                if self.encoding == "label":
                    result[col] = self.encoders[col].inverse_transform(
                        result[col].astype(int)
                    )
                    
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get output feature names after transformation."""
        return self._get_output_columns()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computed statistics from training data."""
        return self._statistics.copy()
    
    def _auto_detect_features(self, data: pd.DataFrame) -> None:
        """Auto-detect numerical and categorical features."""
        for col in data.columns:
            if col == self.target_column:
                continue
            if data[col].dtype in ["int64", "float64"]:
                # Check if actually categorical (few unique values)
                if data[col].nunique() < min(10, len(data) * 0.05):
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
                
        logger.info(f"Auto-detected {len(self.numerical_features)} numerical "
                   f"and {len(self.categorical_features)} categorical features")
    
    def _fit_imputers(self, data: pd.DataFrame) -> None:
        """Fit imputers for missing values."""
        # Numerical imputer
        num_cols = [c for c in self.numerical_features if c in data.columns]
        if num_cols:
            if self.imputation == "knn":
                self.imputers["numerical"] = KNNImputer(n_neighbors=5)
            else:
                strategy = "most_frequent" if self.imputation == "mode" else self.imputation
                self.imputers["numerical"] = SimpleImputer(strategy=strategy)
            self.imputers["numerical"].fit(data[num_cols])
            
        # Categorical imputer
        cat_cols = [c for c in self.categorical_features if c in data.columns]
        if cat_cols:
            self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
            self.imputers["categorical"].fit(data[cat_cols])
    
    def _fit_scalers(self, data: pd.DataFrame) -> None:
        """Fit scalers for numerical features."""
        for col in self.numerical_features:
            if col not in data.columns:
                continue
                
            if self.normalization == "standard":
                self.scalers[col] = StandardScaler()
            elif self.normalization == "minmax":
                self.scalers[col] = MinMaxScaler()
            elif self.normalization == "robust":
                self.scalers[col] = RobustScaler()
            elif self.normalization == "log":
                self.scalers[col] = None  # Log transform doesn't need fitting
            else:
                continue
                
            if self.scalers[col] is not None:
                self.scalers[col].fit(data[[col]])
    
    def _fit_encoders(self, data: pd.DataFrame) -> None:
        """Fit encoders for categorical features."""
        for col in self.categorical_features:
            if col not in data.columns:
                continue
                
            if self.encoding == "label":
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(data[col].astype(str))
            elif self.encoding == "one_hot":
                self.encoders[col] = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown="ignore"
                )
                self.encoders[col].fit(data[[col]])
    
    def _transform_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        result = data.copy()
        
        num_cols = [c for c in self.numerical_features if c in data.columns]
        if num_cols and "numerical" in self.imputers:
            result[num_cols] = self.imputers["numerical"].transform(data[num_cols])
            
        cat_cols = [c for c in self.categorical_features if c in data.columns]
        if cat_cols and "categorical" in self.imputers:
            result[cat_cols] = self.imputers["categorical"].transform(data[cat_cols])
            
        return result
    
    def _transform_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical features."""
        result = data.copy()
        
        for col in self.numerical_features:
            if col not in data.columns or col in self.sensitive_attributes:
                continue
                
            values = result[col].values
            
            if self.outlier_method == "iqr":
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            else:  # zscore
                mean = np.mean(values)
                std = np.std(values)
                lower = mean - self.outlier_threshold * std
                upper = mean + self.outlier_threshold * std
                
            result[col] = np.clip(values, lower, upper)
            
        return result
    
    def _transform_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        result = data.copy()
        
        for col in self.numerical_features:
            if col not in data.columns or col not in self.scalers:
                continue
                
            if self.scalers[col] is not None:
                result[col] = self.scalers[col].transform(data[[col]]).ravel()
            elif self.normalization == "log":
                # Log transform for positive values
                values = data[col].values
                min_val = values.min()
                if min_val <= 0:
                    values = values - min_val + 1
                result[col] = np.log1p(values)
                
        return result
    
    def _transform_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        result = data.copy()
        
        for col in self.categorical_features:
            if col not in data.columns or col not in self.encoders:
                continue
                
            if self.encoding == "label":
                # Handle unseen labels
                encoder = self.encoders[col]
                known_classes = set(encoder.classes_)
                values = data[col].astype(str).apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                result[f"{col}_encoded"] = encoder.transform(values)
                result = result.drop(columns=[col])
            elif self.encoding == "one_hot":
                encoder = self.encoders[col]
                encoded = encoder.transform(data[[col]])
                encoded_cols = [f"{col}_{c}" for c in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)
                result = pd.concat([result, encoded_df], axis=1)
                result = result.drop(columns=[col])
                
        return result
    
    def _get_output_columns(self) -> List[str]:
        """Get output column names after transformation."""
        columns = []
        
        for col in self.numerical_features:
            columns.append(col)
            
        for col in self.categorical_features:
            if self.encoding == "label":
                columns.append(f"{col}_encoded")
            elif self.encoding == "one_hot" and col in self.encoders:
                columns.extend(
                    [f"{col}_{c}" for c in self.encoders[col].categories_[0]]
                )
                
        return columns
    
    def _compute_statistics(self, data: pd.DataFrame) -> None:
        """Compute and store statistics from training data."""
        self._statistics = {
            "n_samples": len(data),
            "n_features": len(self.numerical_features) + len(self.categorical_features),
            "numerical_stats": {},
            "categorical_stats": {},
            "missing_rates": {}
        }
        
        for col in self.numerical_features:
            if col in data.columns:
                self._statistics["numerical_stats"][col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "median": data[col].median()
                }
                
        for col in self.categorical_features:
            if col in data.columns:
                self._statistics["categorical_stats"][col] = {
                    "n_unique": data[col].nunique(),
                    "unique_values": data[col].unique().tolist()[:20]
                }
                
        for col in data.columns:
            self._statistics["missing_rates"][col] = data[col].isnull().mean()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save preprocessor state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": {
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "target_column": self.target_column,
                "sensitive_attributes": self.sensitive_attributes,
                "normalization": self.normalization,
                "encoding": self.encoding,
                "imputation": self.imputation,
                "handle_outliers": self.handle_outliers,
                "outlier_method": self.outlier_method,
                "outlier_threshold": self.outlier_threshold,
            },
            "statistics": self._statistics,
            "feature_order": self._feature_order,
            "is_fitted": self._is_fitted
        }
        
        # Save scalers
        import joblib
        scalers_path = path.with_suffix(".scalers.pkl")
        joblib.dump(self.scalers, scalers_path)
        
        # Save encoders
        encoders_path = path.with_suffix(".encoders.pkl")
        joblib.dump(self.encoders, encoders_path)
        
        # Save imputers
        imputers_path = path.with_suffix(".imputers.pkl")
        joblib.dump(self.imputers, imputers_path)
        
        # Save config
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TabularPreprocessor":
        """Load preprocessor state."""
        import joblib
        
        path = Path(path)
        
        with open(path.with_suffix(".json"), "r") as f:
            state = json.load(f)
            
        preprocessor = cls(**state["config"])
        preprocessor._statistics = state["statistics"]
        preprocessor._feature_order = state["feature_order"]
        preprocessor._is_fitted = state["is_fitted"]
        
        # Load scalers
        scalers_path = path.with_suffix(".scalers.pkl")
        if scalers_path.exists():
            preprocessor.scalers = joblib.load(scalers_path)
            
        # Load encoders
        encoders_path = path.with_suffix(".encoders.pkl")
        if encoders_path.exists():
            preprocessor.encoders = joblib.load(encoders_path)
            
        # Load imputers
        imputers_path = path.with_suffix(".imputers.pkl")
        if imputers_path.exists():
            preprocessor.imputers = joblib.load(imputers_path)
            
        return preprocessor
