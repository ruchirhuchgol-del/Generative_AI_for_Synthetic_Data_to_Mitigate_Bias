"""
Format Converter
================

Utilities for converting between different data formats.
Handles cross-format transformations and compatibility.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings


class FormatConverter:
    """
    Comprehensive format converter for data transformations.
    
    Supports conversions between:
    - NumPy arrays and Pandas DataFrames
    - Various file formats
    - Different data structures
    - ML framework formats
    """
    
    @staticmethod
    def numpy_to_pandas(
        data: np.ndarray,
        columns: Optional[List[str]] = None,
        index: Optional[np.ndarray] = None,
        dtype: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Convert NumPy array to Pandas DataFrame.
        
        Args:
            data: NumPy array (2D)
            columns: Column names
            index: Index array
            dtype: Column data types
            
        Returns:
            Pandas DataFrame
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if columns is None:
            columns = [f"col_{i}" for i in range(data.shape[1])]
        
        df = pd.DataFrame(data, columns=columns, index=index)
        
        if dtype:
            for col, dt in dtype.items():
                if col in df.columns:
                    df[col] = df[col].astype(dt)
        
        return df
    
    @staticmethod
    def pandas_to_numpy(
        data: pd.DataFrame,
        include_index: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert Pandas DataFrame to NumPy array.
        
        Args:
            data: Pandas DataFrame
            include_index: Whether to include index as first column
            
        Returns:
            NumPy array (and optionally index array)
        """
        if include_index:
            values = data.values
            index = data.index.values
            return np.column_stack([index, values])
        
        return data.values
    
    @staticmethod
    def dict_to_dataframe(
        data: Dict[str, Union[np.ndarray, list]],
        orient: str = "columns"
    ) -> pd.DataFrame:
        """
        Convert dictionary to DataFrame.
        
        Args:
            data: Dictionary of arrays/lists
            orient: Orientation ('columns', 'index', 'records')
            
        Returns:
            Pandas DataFrame
        """
        if orient == "columns":
            # Keys are column names
            return pd.DataFrame(data)
        elif orient == "index":
            # Keys are row indices
            return pd.DataFrame.from_dict(data, orient="index")
        elif orient == "records":
            # List of dictionaries
            return pd.DataFrame.from_records(data)
        else:
            return pd.DataFrame(data)
    
    @staticmethod
    def dataframe_to_dict(
        data: pd.DataFrame,
        orient: str = "list",
        include_index: bool = False
    ) -> Dict[str, Any]:
        """
        Convert DataFrame to dictionary.
        
        Args:
            data: Pandas DataFrame
            orient: Output orientation
                - 'list': {column: [values]}
                - 'series': {column: Series}
                - 'dict': {index: {column: value}}
                - 'records': [{column: value}, ...]
            include_index: Include index in output
            
        Returns:
            Dictionary
        """
        result = data.to_dict(orient=orient)
        
        if include_index:
            result["_index"] = data.index.tolist()
        
        return result
    
    @staticmethod
    def records_to_dataframe(
        records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert list of records to DataFrame.
        
        Args:
            records: List of dictionaries
            
        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame.from_records(records)
    
    @staticmethod
    def dataframe_to_records(
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to list of records.
        
        Args:
            data: Pandas DataFrame
            
        Returns:
            List of dictionaries
        """
        return data.to_dict(orient="records")
    
    @staticmethod
    def csv_to_parquet(
        csv_path: str,
        parquet_path: str,
        compression: str = "snappy",
        chunk_size: Optional[int] = None
    ) -> str:
        """
        Convert CSV to Parquet format.
        
        Args:
            csv_path: Input CSV file path
            parquet_path: Output Parquet file path
            compression: Compression type
            chunk_size: Optional chunk size for large files
            
        Returns:
            Output file path
        """
        if chunk_size:
            # Process in chunks
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path)
        
        df.to_parquet(parquet_path, compression=compression)
        return parquet_path
    
    @staticmethod
    def parquet_to_csv(
        parquet_path: str,
        csv_path: str,
        compression: Optional[str] = None,
        index: bool = False
    ) -> str:
        """
        Convert Parquet to CSV format.
        
        Args:
            parquet_path: Input Parquet file path
            csv_path: Output CSV file path
            compression: Compression type
            index: Whether to include index
            
        Returns:
            Output file path
        """
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=index, compression=compression)
        return csv_path
    
    @staticmethod
    def json_to_dataframe(
        json_path: str,
        orient: str = "records",
        lines: bool = False
    ) -> pd.DataFrame:
        """
        Convert JSON to DataFrame.
        
        Args:
            json_path: Input JSON file path
            orient: JSON orientation
            lines: Whether JSON is line-delimited
            
        Returns:
            Pandas DataFrame
        """
        return pd.read_json(json_path, orient=orient, lines=lines)
    
    @staticmethod
    def dataframe_to_json(
        data: pd.DataFrame,
        json_path: str,
        orient: str = "records",
        indent: int = 2,
        lines: bool = False
    ) -> str:
        """
        Convert DataFrame to JSON.
        
        Args:
            data: Pandas DataFrame
            json_path: Output JSON file path
            orient: JSON orientation
            indent: Indentation level
            lines: Whether to use line-delimited format
            
        Returns:
            Output file path
        """
        data.to_json(json_path, orient=orient, indent=indent, lines=lines)
        return json_path
    
    @staticmethod
    def to_torch_tensor(
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        dtype: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Convert data to PyTorch tensor.
        
        Args:
            data: Input data
            dtype: Target dtype ('float', 'float32', 'float64', 'int', 'int32', 'int64')
            device: Target device
            
        Returns:
            PyTorch tensor (or dict of tensors)
        """
        import torch
        
        dtype_map = {
            "float": torch.float32,
            "float32": torch.float32,
            "float64": torch.float64,
            "int": torch.int32,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        
        torch_dtype = dtype_map.get(dtype, None) if dtype else None
        
        if isinstance(data, dict):
            return {
                k: torch.tensor(v, dtype=torch_dtype, device=device)
                for k, v in data.items()
            }
        elif isinstance(data, pd.DataFrame):
            tensor = torch.tensor(data.values, dtype=torch_dtype, device=device)
        else:
            tensor = torch.tensor(data, dtype=torch_dtype, device=device)
        
        return tensor
    
    @staticmethod
    def from_torch_tensor(
        tensor,
        to_type: str = "numpy",
        columns: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Convert PyTorch tensor to other format.
        
        Args:
            tensor: PyTorch tensor
            to_type: Target type ('numpy', 'pandas')
            columns: Column names for DataFrame
            
        Returns:
            Converted data
        """
        import torch
        
        array = tensor.detach().cpu().numpy()
        
        if to_type == "pandas":
            if columns is None:
                columns = [f"col_{i}" for i in range(array.shape[1])]
            return pd.DataFrame(array, columns=columns)
        
        return array
    
    @staticmethod
    def to_tensorflow(
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        dtype: Optional[str] = None
    ):
        """
        Convert data to TensorFlow tensor.
        
        Args:
            data: Input data
            dtype: Target dtype
            
        Returns:
            TensorFlow tensor (or dict of tensors)
        """
        import tensorflow as tf
        
        if isinstance(data, dict):
            return {
                k: tf.convert_to_tensor(v, dtype=dtype)
                for k, v in data.items()
            }
        elif isinstance(data, pd.DataFrame):
            return tf.convert_to_tensor(data.values, dtype=dtype)
        
        return tf.convert_to_tensor(data, dtype=dtype)
    
    @staticmethod
    def from_tensorflow(
        tensor,
        to_type: str = "numpy",
        columns: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Convert TensorFlow tensor to other format.
        
        Args:
            tensor: TensorFlow tensor
            to_type: Target type ('numpy', 'pandas')
            columns: Column names for DataFrame
            
        Returns:
            Converted data
        """
        array = tensor.numpy()
        
        if to_type == "pandas":
            if columns is None:
                columns = [f"col_{i}" for i in range(array.shape[1])]
            return pd.DataFrame(array, columns=columns)
        
        return array


class MultiModalConverter:
    """
    Converter for multimodal data structures.
    
    Handles conversions between different modalities and unified formats.
    """
    
    @staticmethod
    def merge_modalities(
        modalities: Dict[str, np.ndarray],
        align_on: str = "samples"
    ) -> np.ndarray:
        """
        Merge multiple modalities into single array.
        
        Args:
            modalities: Dictionary of modality arrays
            align_on: How to align ('samples', 'features')
            
        Returns:
            Merged array
        """
        arrays = list(modalities.values())
        
        if align_on == "samples":
            # Concatenate along feature dimension
            return np.concatenate(arrays, axis=1)
        else:
            # Stack along sample dimension
            return np.concatenate(arrays, axis=0)
    
    @staticmethod
    def split_modalities(
        data: np.ndarray,
        split_points: List[int],
        modality_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split single array into modalities.
        
        Args:
            data: Combined data array
            split_points: Column indices where to split
            modality_names: Names for each modality
            
        Returns:
            Dictionary of modality arrays
        """
        modality_names = modality_names or [
            f"modality_{i}" for i in range(len(split_points) + 1)
        ]
        
        result = {}
        prev = 0
        
        for i, point in enumerate(split_points):
            result[modality_names[i]] = data[:, prev:point]
            prev = point
        
        result[modality_names[-1]] = data[:, prev:]
        
        return result
    
    @staticmethod
    def align_modalities(
        modalities: Dict[str, np.ndarray],
        reference: str,
        method: str = "truncate"
    ) -> Dict[str, np.ndarray]:
        """
        Align modalities to same number of samples.
        
        Args:
            modalities: Dictionary of modality arrays
            reference: Reference modality name
            method: Alignment method ('truncate', 'pad', 'resample')
            
        Returns:
            Aligned modality dictionary
        """
        ref_size = len(modalities[reference])
        aligned = {}
        
        for name, data in modalities.items():
            if len(data) == ref_size:
                aligned[name] = data
            elif len(data) > ref_size:
                if method == "truncate":
                    aligned[name] = data[:ref_size]
                elif method == "resample":
                    indices = np.linspace(0, len(data) - 1, ref_size).astype(int)
                    aligned[name] = data[indices]
            else:
                if method == "pad":
                    pad_size = ref_size - len(data)
                    aligned[name] = np.pad(data, ((0, pad_size), (0, 0)))
                elif method == "resample":
                    indices = np.linspace(0, len(data) - 1, ref_size).astype(int)
                    aligned[name] = data[indices]
        
        return aligned


class SparseConverter:
    """
    Converter for sparse matrix formats.
    
    Handles conversions between dense and sparse representations.
    """
    
    @staticmethod
    def to_sparse(
        data: np.ndarray,
        format: str = "csr",
        threshold: float = 0.0
    ):
        """
        Convert dense array to sparse matrix.
        
        Args:
            data: Dense array
            format: Sparse format ('csr', 'csc', 'coo')
            threshold: Values below this set to zero
            
        Returns:
            Sparse matrix
        """
        from scipy import sparse
        
        if threshold > 0:
            data = np.where(np.abs(data) < threshold, 0, data)
        
        if format == "csr":
            return sparse.csr_matrix(data)
        elif format == "csc":
            return sparse.csc_matrix(data)
        elif format == "coo":
            return sparse.coo_matrix(data)
        
        return sparse.csr_matrix(data)
    
    @staticmethod
    def from_sparse(
        sparse_matrix,
        to_type: str = "numpy"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Convert sparse matrix to dense format.
        
        Args:
            sparse_matrix: Sparse matrix
            to_type: Target type ('numpy', 'pandas')
            
        Returns:
            Dense array or DataFrame
        """
        dense = sparse_matrix.toarray()
        
        if to_type == "pandas":
            return pd.DataFrame(dense)
        
        return dense


class BatchConverter:
    """
    Converter for batch processing of large datasets.
    """
    
    def __init__(
        self,
        batch_size: int = 10000,
        output_dir: str = "data/converted"
    ):
        """
        Initialize batch converter.
        
        Args:
            batch_size: Number of samples per batch
            output_dir: Output directory
        """
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_large_csv(
        self,
        input_path: str,
        output_format: str = "parquet",
        output_filename: str = "converted"
    ) -> str:
        """
        Convert large CSV file in batches.
        
        Args:
            input_path: Input CSV path
            output_format: Output format
            output_filename: Output filename
            
        Returns:
            Output file path
        """
        output_path = self.output_dir / f"{output_filename}.{output_format}"
        
        # Process in chunks
        chunks = []
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=self.batch_size)):
            chunks.append(chunk)
        
        # Combine and save
        df = pd.concat(chunks, ignore_index=True)
        
        if output_format == "parquet":
            df.to_parquet(output_path)
        elif output_format == "feather":
            df.to_feather(output_path)
        elif output_format == "hdf":
            df.to_hdf(output_path, key="data")
        else:
            df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def convert_in_batches(
        self,
        data: np.ndarray,
        converter_func,
        n_batches: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply conversion function in batches.
        
        Args:
            data: Input data
            converter_func: Conversion function
            n_batches: Number of batches (auto if None)
            
        Returns:
            Converted data
        """
        if n_batches is None:
            n_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        results = []
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(data))
            
            batch = data[start:end]
            converted = converter_func(batch)
            results.append(converted)
        
        return np.concatenate(results, axis=0)
