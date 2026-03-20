RATIO   Training split ratio (default: 0.7)
    --val-split RATIO     Validation split ratio (default: 0.15)
    --seed N              Random seed
    -h, --help            Show this help message

Examples:
    python preprocess_raw_data.py --input data/raw/dataset.csv --output data/processed/
    python preprocess_raw_data.py --input data/raw/ --sensitive-attrs gender,race --balance
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Raw Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data file or directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Schema file for data validation"
    )
    
    parser.add_argument(
        "--sensitive-attrs",
        type=str,
        default=None,
        help="Comma-separated list of sensitive attributes"
    )
    
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["standard", "minmax", "robust", "none"],
        default="standard",
        help="Normalization method for numerical features"
    )
    
    parser.add_argument(
        "--encode-categorical",
        type=str,
        choices=["onehot", "label", "target", "ordinal"],
        default="onehot",
        help="Encoding method for categorical features"
    )
    
    parser.add_argument(
        "--handle-missing",
        type=str,
        choices=["drop", "mean", "median", "mode", "knn", "constant"],
        default="mean",
        help="Missing value handling strategy"
    )
    
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance dataset across sensitive groups"
    )
    
    parser.add_argument(
        "--balance-method",
        type=str,
        choices=["oversample", "undersample", "smote"],
        default="oversample",
        help="Method for balancing sensitive groups"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training split ratio"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Target column name for supervised learning"
    )
    
    parser.add_argument(
        "--save-transformers",
        action="store_true",
        default=True,
        help="Save fitted transformers for inference"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information"
    )
    
    return parser.parse_args()


class DataPreprocessor:
    """
    Data preprocessor for fair synthetic data generation.
    
    This class handles:
    - Data loading and validation
    - Missing value imputation
    - Feature encoding and normalization
    - Sensitive attribute handling
    - Train/val/test splitting
    - Dataset balancing
    """
    
    def __init__(
        self,
        sensitive_attrs: Optional[List[str]] = None,
        normalize: str = "standard",
        encode_categorical: str = "onehot",
        handle_missing: str = "mean",
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            sensitive_attrs: List of sensitive attribute column names
            normalize: Normalization method
            encode_categorical: Categorical encoding method
            handle_missing: Missing value handling strategy
            seed: Random seed
            verbose: Whether to print progress information
        """
        self.sensitive_attrs = sensitive_attrs or []
        self.normalize = normalize
        self.encode_categorical = encode_categorical
        self.handle_missing = handle_missing
        self.seed = seed
        self.verbose = verbose
        
        # Fitted transformers
        self.scalers_: Dict[str, Any] = {}
        self.encoders_: Dict[str, Any] = {}
        self.imputers_: Dict[str, Any] = {}
        
        # Column info
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.binary_cols_: List[str] = []
        
        np.random.seed(seed)
    
    def log(self, msg: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Preprocessor] {msg}")
    
    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load data from file or directory.
        
        Args:
            input_path: Path to input data file or directory
            
        Returns:
            Loaded DataFrame
        """
        path = Path(input_path)
        
        if path.is_file():
            self.log(f"Loading data from {path}")
            
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".json":
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
                
        elif path.is_dir():
            self.log(f"Loading data from directory {path}")
            # Load all CSV files in directory
            files = list(path.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        self.log(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Infer column types from data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numerical_cols, categorical_cols, binary_cols)
        """
        numerical = []
        categorical = []
        binary = []
        
        for col in df.columns:
            if col in self.sensitive_attrs:
                # Treat sensitive attributes as categorical for encoding
                categorical.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                n_unique = df[col].nunique()
                if n_unique == 2:
                    binary.append(col)
                else:
                    numerical.append(col)
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                n_unique = df[col].nunique()
                if n_unique == 2:
                    binary.append(col)
                else:
                    categorical.append(col)
            else:
                categorical.append(col)
        
        return numerical, categorical, binary
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit imputers or use existing ones
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if self.handle_missing == "drop":
            initial_len = len(df)
            df = df.dropna()
            self.log(f"Dropped {initial_len - len(df)} rows with missing values")
            return df
        
        for col in df.columns:
            if df[col].isna().any():
                if fit:
                    if df[col].dtype in ['int64', 'float64']:
                        if self.handle_missing == "mean":
                            imputer = SimpleImputer(strategy="mean")
                        elif self.handle_missing == "median":
                            imputer = SimpleImputer(strategy="median")
                        elif self.handle_missing == "knn":
                            imputer = KNNImputer(n_neighbors=5)
                        else:
                            imputer = SimpleImputer(strategy="mean")
                    else:
                        imputer = SimpleImputer(strategy="most_frequent")
                    
                    self.imputers_[col] = imputer
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                else:
                    if col in self.imputers_:
                        df[col] = self.imputers_[col].transform(df[[col]]).ravel()
        
        self.log(f"Missing values handled using '{self.handle_missing}' strategy")
        return df
    
    def encode_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            target_col: Target column (for target encoding)
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for col in self.categorical_cols_:
            if col == target_col:
                continue
                
            if self.encode_categorical == "onehot":
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    self.encoders_[col] = encoder
                else:
                    encoder = self.encoders_[col]
                    encoded = encoder.transform(df[[col]])
                
                # Create new column names
                categories = encoder.categories_[0]
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in categories],
                    index=df.index
                )
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                
            elif self.encode_categorical == "label":
                if fit:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders_[col] = encoder
                else:
                    df[col] = self.encoders_[col].transform(df[col].astype(str))
                    
            elif self.encode_categorical == "target":
                if target_col is None:
                    # Fall back to label encoding if no target
                    if fit:
                        encoder = LabelEncoder()
                        df[col] = encoder.fit_transform(df[col].astype(str))
                        self.encoders_[col] = encoder
                    else:
                        df[col] = self.encoders_[col].transform(df[col].astype(str))
                else:
                    if fit:
                        encoder = TargetEncoder()
                        df[col] = encoder.fit_transform(
                            df[[col]].astype(str),
                            df[target_col]
                        )
                        self.encoders_[col] = encoder
                    else:
                        df[col] = self.encoders_[col].transform(df[[col]].astype(str))
        
        self.log(f"Encoded categorical features using '{self.encode_categorical}' method")
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers or use existing ones
            
        Returns:
            DataFrame with normalized features
        """
        if self.normalize == "none":
            return df
        
        df = df.copy()
        
        for col in self.numerical_cols_:
            if col not in df.columns:
                continue
                
            if fit:
                if self.normalize == "standard":
                    scaler = StandardScaler()
                elif self.normalize == "minmax":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                self.scalers_[col] = scaler
                df[col] = scaler.fit_transform(df[[col]])
            else:
                if col in self.scalers_:
                    df[col] = self.scalers_[col].transform(df[[col]])
        
        self.log(f"Normalized numerical features using '{self.normalize}' method")
        return df
    
    def balance_dataset(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Balance dataset across sensitive groups.
        
        Args:
            df: Input DataFrame
            target_col: Target column for stratified balancing
            
        Returns:
            Balanced DataFrame
        """
        if not self.sensitive_attrs:
            self.log("No sensitive attributes specified, skipping balancing")
            return df
        
        # Create group identifier
        group_col = "_group_id"
        df[group_col] = df[self.sensitive_attrs].astype(str).agg('_'.join, axis=1)
        
        # Get group sizes
        group_sizes = df[group_col].value_counts()
        max_size = group_sizes.max()
        min_size = group_sizes.min()
        
        self.log(f"Group sizes before balancing: min={min_size}, max={max_size}")
        
        if self.balance_method == "oversample":
            balanced_dfs = []
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                if len(group_df) < max_size:
                    # Oversample to match max size
                    group_df = group_df.sample(
                        n=max_size,
                        replace=True,
                        random_state=self.seed
                    )
                balanced_dfs.append(group_df)
            df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif self.balance_method == "undersample":
            balanced_dfs = []
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                if len(group_df) > min_size:
                    group_df = group_df.sample(
                        n=min_size,
                        replace=False,
                        random_state=self.seed
                    )
                balanced_dfs.append(group_df)
            df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Drop temporary column
        df = df.drop(group_col, axis=1)
        
        self.log(f"Balanced dataset: {len(df)} rows using '{self.balance_method}' method")
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Preprocessed DataFrame
        """
        self.log("Starting preprocessing pipeline...")
        
        # Infer column types
        self.numerical_cols_, self.categorical_cols_, self.binary_cols_ = \
            self.infer_column_types(df)
        
        self.log(f"Inferred {len(self.numerical_cols_)} numerical, "
                f"{len(self.categorical_cols_)} categorical, "
                f"{len(self.binary_cols_)} binary columns")
        
        # Handle missing values
        df = self.handle_missing_values(df, fit=True)
        
        # Encode categorical features
        df = self.encode_features(df, target_col=target_col, fit=True)
        
        # Normalize numerical features
        df = self.normalize_features(df, fit=True)
        
        self.log("Preprocessing complete!")
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        df = self.handle_missing_values(df, fit=False)
        
        # Encode categorical features
        df = self.encode_features(df, target_col=target_col, fit=False)
        
        # Normalize numerical features
        df = self.normalize_features(df, fit=False)
        
        return df
    
    def save_transformers(self, output_dir: str):
        """
        Save fitted transformers to disk.
        
        Args:
            output_dir: Output directory
        """
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save imputers
        for col, imputer in self.imputers_.items():
            joblib.dump(imputer, output_path / f"imputer_{col}.joblib")
        
        # Save encoders
        for col, encoder in self.encoders_.items():
            joblib.dump(encoder, output_path / f"encoder_{col}.joblib")
        
        # Save scalers
        for col, scaler in self.scalers_.items():
            joblib.dump(scaler, output_path / f"scaler_{col}.joblib")
        
        # Save metadata
        metadata = {
            "sensitive_attrs": self.sensitive_attrs,
            "numerical_cols": self.numerical_cols_,
            "categorical_cols": self.categorical_cols_,
            "binary_cols": self.binary_cols_,
            "normalize": self.normalize,
            "encode_categorical": self.encode_categorical,
            "handle_missing": self.handle_missing,
            "seed": self.seed
        }
        
        with open(output_path / "preprocessor_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.log(f"Saved transformers to {output_path}")


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sensitive_attrs: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        sensitive_attrs: Sensitive attributes for stratification
        target_col: Target column for stratification
        seed: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Create stratification column
    if sensitive_attrs and target_col:
        stratify = df[sensitive_attrs + [target_col]].astype(str).agg('_'.join, axis=1)
    elif target_col:
        stratify = df[target_col]
    else:
        stratify = None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=stratify,
        random_state=seed
    )
    
    # Second split: val vs test
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    
    if stratify is not None:
        # Recreate stratify for temp_df
        temp_stratify = temp_df.index.map(lambda i: stratify.iloc[df.index.get_loc(i)])
    else:
        temp_stratify = None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adj),
        stratify=temp_stratify,
        random_state=seed
    )
    
    return train_df, val_df, test_df


def generate_preprocessing_report(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    sensitive_attrs: List[str],
    output_path: str
):
    """
    Generate a preprocessing report.
    
    Args:
        original_df: Original DataFrame
        processed_df: Processed DataFrame
        sensitive_attrs: List of sensitive attributes
        output_path: Path to save report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "original_shape": list(original_df.shape),
        "processed_shape": list(processed_df.shape),
        "columns_added": list(set(processed_df.columns) - set(original_df.columns)),
        "columns_removed": list(set(original_df.columns) - set(processed_df.columns)),
        "sensitive_attributes": sensitive_attrs,
        "missing_values_before": original_df.isna().sum().to_dict(),
        "missing_values_after": processed_df.isna().sum().to_dict(),
        "dtypes_before": {k: str(v) for k, v in original_df.dtypes.to_dict().items()},
        "dtypes_after": {k: str(v) for k, v in processed_df.dtypes.to_dict().items()}
    }
    
    # Add sensitive attribute statistics
    report["sensitive_stats"] = {}
    for attr in sensitive_attrs:
        if attr in original_df.columns:
            report["sensitive_stats"][attr] = {
                "unique_values": original_df[attr].nunique(),
                "value_counts": original_df[attr].value_counts().to_dict()
            }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 60)
    print("Fair Synthetic Data Generator - Data Preprocessing")
    print("=" * 60)
    
    # Parse sensitive attributes
    sensitive_attrs = None
    if args.sensitive_attrs:
        sensitive_attrs = [s.strip() for s in args.sensitive_attrs.split(",")]
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        sensitive_attrs=sensitive_attrs,
        normalize=args.normalize,
        encode_categorical=args.encode_categorical,
        handle_missing=args.handle_missing,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Load data
    df = preprocessor.load_data(args.input)
    original_df = df.copy()
    
    # Preprocess
    df_processed = preprocessor.fit_transform(df, target_col=args.target_column)
    
    # Balance if requested
    if args.balance:
        df_processed = preprocessor.balance_dataset(df_processed, target_col=args.target_column)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_df, val_df, test_df = split_data(
        df_processed,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        sensitive_attrs=sensitive_attrs,
        target_col=args.target_column,
        seed=args.seed
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Save processed data
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"\nSaved processed data to {output_dir}")
    
    # Save transformers
    if args.save_transformers:
        preprocessor.save_transformers(str(output_dir / "transformers"))
    
    # Generate report
    generate_preprocessing_report(
        original_df,
        df_processed,
        sensitive_attrs or [],
        str(output_dir / "preprocessing_report.json")
    )
    
    print(f"\nPreprocessing complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
