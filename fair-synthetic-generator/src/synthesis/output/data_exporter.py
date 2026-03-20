"""
Data Exporter
=============

Comprehensive data export utilities for synthetic data.
Supports multiple formats and provides metadata tracking.
"""

from typing import Any, Dict, List, Optional, Union
import json
import os
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


class DataExporter:
    """
    Comprehensive data exporter for synthetic data.
    
    Supports multiple output formats with metadata tracking:
    - CSV (with optional compression)
    - JSON (various orientations)
    - Parquet (efficient columnar format)
    - HDF5 (hierarchical data format)
    - Arrow (Apache Arrow format)
    - SQL databases
    - Feather (fast binary format)
    """
    
    SUPPORTED_FORMATS = ["csv", "json", "parquet", "hdf5", "arrow", "feather", "sql"]
    
    def __init__(
        self,
        output_dir: str = "data/synthetic",
        compression: Optional[str] = None,
        create_metadata: bool = True,
        timestamp_outputs: bool = False
    ):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Default output directory
            compression: Default compression format ('gzip', 'bz2', 'xz', 'zip')
            create_metadata: Whether to create metadata files
            timestamp_outputs: Whether to add timestamps to filenames
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.create_metadata = create_metadata
        self.timestamp_outputs = timestamp_outputs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export history
        self._export_history = []
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _prepare_filename(
        self,
        filename: str,
        format: str
    ) -> Path:
        """Prepare full file path with optional timestamp."""
        if self.timestamp_outputs:
            base, ext = os.path.splitext(filename)
            if not ext:
                ext = f".{format}"
            filename = f"{base}_{self._get_timestamp()}{ext}"
        elif not os.path.splitext(filename)[1]:
            filename = f"{filename}.{format}"
        
        return self.output_dir / filename
    
    def _to_dataframe(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            # Handle dictionary of arrays
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"feature_{i}" for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def export_csv(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        filename: str,
        columns: Optional[List[str]] = None,
        index: bool = False,
        encoding: str = "utf-8",
        **kwargs
    ) -> str:
        """
        Export to CSV format.
        
        Args:
            data: Data to export
            filename: Output filename
            columns: Column names (for numpy arrays)
            index: Whether to include index
            encoding: File encoding
            **kwargs: Additional pandas to_csv arguments
            
        Returns:
            Output file path
        """
        df = self._to_dataframe(data, columns)
        path = self._prepare_filename(filename, "csv")
        
        # Add compression extension if specified
        if self.compression:
            path = Path(str(path) + f".{self.compression}")
        
        df.to_csv(
            path,
            index=index,
            encoding=encoding,
            compression=self.compression,
            **kwargs
        )
        
        self._record_export("csv", path, len(df))
        return str(path)
    
    def export_json(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict, List],
        filename: str,
        orient: str = "records",
        indent: Optional[int] = 2,
        **kwargs
    ) -> str:
        """
        Export to JSON format.
        
        Args:
            data: Data to export
            filename: Output filename
            orient: JSON orientation ('records', 'index', 'columns', 'values', 'table')
            indent: Indentation level
            **kwargs: Additional arguments
            
        Returns:
            Output file path
        """
        path = self._prepare_filename(filename, "json")
        
        # Convert data to JSON-serializable format
        if isinstance(data, np.ndarray):
            json_data = data.tolist()
        elif isinstance(data, pd.DataFrame):
            json_data = json.loads(data.to_json(orient=orient, **kwargs))
        elif isinstance(data, dict):
            json_data = self._convert_numpy(data)
        else:
            json_data = data
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=indent, default=str)
        
        self._record_export("json", path, len(json_data) if hasattr(json_data, '__len__') else 1)
        return str(path)
    
    def _convert_numpy(self, obj: Any) -> Any:
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        return obj
    
    def export_parquet(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        filename: str,
        columns: Optional[List[str]] = None,
        engine: str = "auto",
        compression: str = "snappy",
        **kwargs
    ) -> str:
        """
        Export to Parquet format.
        
        Args:
            data: Data to export
            filename: Output filename
            columns: Column names
            engine: Parquet engine ('auto', 'pyarrow', 'fastparquet')
            compression: Compression type ('snappy', 'gzip', 'brotli', 'none')
            **kwargs: Additional arguments
            
        Returns:
            Output file path
        """
        df = self._to_dataframe(data, columns)
        path = self._prepare_filename(filename, "parquet")
        
        df.to_parquet(
            path,
            engine=engine,
            compression=compression,
            **kwargs
        )
        
        self._record_export("parquet", path, len(df))
        return str(path)
    
    def export_hdf5(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        filename: str,
        key: str = "data",
        mode: str = "w",
        compression: str = "gzip",
        **kwargs
    ) -> str:
        """
        Export to HDF5 format.
        
        Args:
            data: Data to export
            filename: Output filename
            key: HDF5 key for the dataset
            mode: File mode ('w', 'a', 'r+')
            compression: Compression type
            **kwargs: Additional arguments
            
        Returns:
            Output file path
        """
        import h5py
        
        path = self._prepare_filename(filename, "h5")
        
        with h5py.File(path, mode) as f:
            if isinstance(data, dict):
                # Multiple datasets
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        f.create_dataset(k, data=v, compression=compression, **kwargs)
            elif isinstance(data, pd.DataFrame):
                # DataFrame as single dataset
                f.create_dataset(key, data=data.values, compression=compression, **kwargs)
                # Store column names as attribute
                f[key].attrs['columns'] = list(data.columns)
            elif isinstance(data, np.ndarray):
                # Single array
                f.create_dataset(key, data=data, compression=compression, **kwargs)
        
        n_samples = len(data) if hasattr(data, '__len__') else 1
        self._record_export("hdf5", path, n_samples)
        return str(path)
    
    def export_arrow(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        filename: str,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Export to Apache Arrow format.
        
        Args:
            data: Data to export
            filename: Output filename
            columns: Column names
            **kwargs: Additional arguments
            
        Returns:
            Output file path
        """
        import pyarrow as pa
        import pyarrow.feather as feather
        
        df = self._to_dataframe(data, columns)
        path = self._prepare_filename(filename, "arrow")
        
        table = pa.Table.from_pandas(df)
        feather.write_feather(table, path, **kwargs)
        
        self._record_export("arrow", path, len(df))
        return str(path)
    
    def export_feather(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        filename: str,
        columns: Optional[List[str]] = None,
        compression: Optional[str] = "zstd",
        **kwargs
    ) -> str:
        """
        Export to Feather format.
        
        Args:
            data: Data to export
            filename: Output filename
            columns: Column names
            compression: Compression type ('zstd', 'lz4', 'uncompressed')
            **kwargs: Additional arguments
            
        Returns:
            Output file path
        """
        df = self._to_dataframe(data, columns)
        path = self._prepare_filename(filename, "feather")
        
        df.to_feather(path, compression=compression, **kwargs)
        
        self._record_export("feather", path, len(df))
        return str(path)
    
    def export_sql(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        table_name: str,
        connection: str,
        columns: Optional[List[str]] = None,
        if_exists: str = "replace",
        index: bool = False,
        **kwargs
    ) -> str:
        """
        Export to SQL database.
        
        Args:
            data: Data to export
            table_name: Target table name
            connection: Database connection string or SQLAlchemy engine
            columns: Column names
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write index
            **kwargs: Additional arguments
            
        Returns:
            Table name
        """
        from sqlalchemy import create_engine
        
        df = self._to_dataframe(data, columns)
        
        # Create engine if connection is string
        if isinstance(connection, str):
            engine = create_engine(connection)
        else:
            engine = connection
        
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=index,
            **kwargs
        )
        
        self._record_export("sql", f"{connection}:{table_name}", len(df))
        return table_name
    
    def export(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        filename: str,
        format: str = "csv",
        **kwargs
    ) -> str:
        """
        Export data to specified format.
        
        Args:
            data: Data to export
            filename: Output filename (extension optional)
            format: Output format ('csv', 'json', 'parquet', 'hdf5', 'arrow', 'feather', 'sql')
            **kwargs: Format-specific arguments
            
        Returns:
            Output file path
        """
        format = format.lower()
        
        if format == "csv":
            return self.export_csv(data, filename, **kwargs)
        elif format == "json":
            return self.export_json(data, filename, **kwargs)
        elif format == "parquet":
            return self.export_parquet(data, filename, **kwargs)
        elif format in ["hdf5", "h5"]:
            return self.export_hdf5(data, filename, **kwargs)
        elif format == "arrow":
            return self.export_arrow(data, filename, **kwargs)
        elif format == "feather":
            return self.export_feather(data, filename, **kwargs)
        elif format == "sql":
            return self.export_sql(data, filename, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")
    
    def _record_export(
        self,
        format: str,
        path: str,
        n_records: int
    ) -> None:
        """Record export in history."""
        self._export_history.append({
            "timestamp": datetime.now().isoformat(),
            "format": format,
            "path": str(path),
            "n_records": n_records,
        })
    
    def get_export_history(self) -> List[Dict[str, Any]]:
        """Get export history."""
        return self._export_history.copy()
    
    def clear_history(self) -> None:
        """Clear export history."""
        self._export_history = []


class MetadataWriter:
    """
    Writes comprehensive metadata for synthetic datasets.
    
    Includes generation parameters, statistics, and quality metrics.
    """
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize metadata writer.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_metadata(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        filename: str,
        generation_params: Optional[Dict] = None,
        fairness_metrics: Optional[Dict] = None,
        quality_scores: Optional[Dict] = None,
        column_info: Optional[Dict] = None,
        **extra_fields
    ) -> str:
        """
        Write comprehensive metadata file.
        
        Args:
            data: Synthetic data
            filename: Output filename
            generation_params: Generation parameters
            fairness_metrics: Fairness evaluation metrics
            quality_scores: Quality assessment scores
            column_info: Column metadata
            **extra_fields: Additional metadata fields
            
        Returns:
            Output file path
        """
        # Compute basic statistics
        if isinstance(data, np.ndarray):
            stats = self._compute_numpy_stats(data)
        elif isinstance(data, pd.DataFrame):
            stats = self._compute_dataframe_stats(data)
        else:
            stats = {}
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "data": stats,
            "generation": generation_params or {},
            "fairness": fairness_metrics or {},
            "quality": quality_scores or {},
            "columns": column_info or {},
            **extra_fields,
        }
        
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(path)
    
    def _compute_numpy_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for numpy array."""
        return {
            "n_samples": int(data.shape[0]),
            "n_features": int(data.shape[1]) if data.ndim > 1 else 1,
            "dtype": str(data.dtype),
            "memory_mb": float(data.nbytes / (1024 * 1024)),
            "nan_count": int(np.isnan(data).sum()),
            "statistics": {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "median": float(np.nanmedian(data)),
            },
        }
    
    def _compute_dataframe_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for DataFrame."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        stats = {
            "n_samples": int(len(data)),
            "n_features": int(len(data.columns)),
            "columns": list(data.columns),
            "dtypes": {col: str(dt) for col, dt in data.dtypes.items()},
            "memory_mb": float(data.memory_usage(deep=True).sum() / (1024 * 1024)),
            "nan_counts": {col: int(data[col].isna().sum()) for col in data.columns},
        }
        
        # Numeric statistics
        if len(numeric_cols) > 0:
            stats["numeric_statistics"] = {
                col: {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "median": float(data[col].median()),
                }
                for col in numeric_cols
            }
        
        return stats
    
    def write_schema(
        self,
        schema: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Write data schema.
        
        Args:
            schema: Schema definition
            filename: Output filename
            
        Returns:
            Output file path
        """
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, default=str)
        
        return str(path)
    
    def write_data_dictionary(
        self,
        columns: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        types: Optional[Dict[str, str]] = None,
        sensitive_columns: Optional[List[str]] = None,
        filename: str = "data_dictionary.json"
    ) -> str:
        """
        Write data dictionary.
        
        Args:
            columns: List of column names
            descriptions: Column descriptions
            types: Column types
            sensitive_columns: List of sensitive columns
            filename: Output filename
            
        Returns:
            Output file path
        """
        descriptions = descriptions or {}
        types = types or {}
        sensitive_columns = sensitive_columns or []
        
        dictionary = {
            "columns": [
                {
                    "name": col,
                    "description": descriptions.get(col, ""),
                    "type": types.get(col, "unknown"),
                    "is_sensitive": col in sensitive_columns,
                }
                for col in columns
            ],
            "sensitive_columns": sensitive_columns,
            "created_at": datetime.now().isoformat(),
        }
        
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2)
        
        return str(path)


class SyntheticDataPackage:
    """
    Creates complete synthetic data packages with documentation.
    
    Packages include:
    - Data files in multiple formats
    - Metadata
    - Schema/data dictionary
    - README documentation
    - Fairness report
    """
    
    def __init__(
        self,
        output_dir: str = "data/synthetic",
        package_name: str = "synthetic_data"
    ):
        """
        Initialize package creator.
        
        Args:
            output_dir: Base output directory
            package_name: Name of the package
        """
        self.output_dir = Path(output_dir)
        self.package_name = package_name
        self.package_dir = self.output_dir / package_name
        
        self.exporter = DataExporter(str(self.package_dir))
        self.metadata_writer = MetadataWriter(str(self.package_dir))
        
        # Create package directory
        self.package_dir.mkdir(parents=True, exist_ok=True)
    
    def create_package(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        schema: Optional[Dict] = None,
        generation_params: Optional[Dict] = None,
        fairness_report: Optional[Dict] = None,
        quality_report: Optional[Dict] = None,
        formats: List[str] = ["csv", "parquet"],
        columns: Optional[List[str]] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
        sensitive_columns: Optional[List[str]] = None,
        license: str = "CC BY 4.0"
    ) -> Dict[str, str]:
        """
        Create complete data package.
        
        Args:
            data: Synthetic data
            schema: Data schema
            generation_params: Generation parameters
            fairness_report: Fairness evaluation
            quality_report: Quality assessment
            formats: Output formats
            columns: Column names
            column_descriptions: Column descriptions
            sensitive_columns: Sensitive attribute columns
            license: License identifier
            
        Returns:
            Dictionary of created files
        """
        files = {}
        
        # 1. Export data in specified formats
        for fmt in formats:
            filename = f"data.{fmt}"
            path = self.exporter.export(data, filename, format=fmt, columns=columns)
            files[f"data_{fmt}"] = path
        
        # 2. Write metadata
        metadata_path = self.metadata_writer.write_metadata(
            data,
            "metadata.json",
            generation_params=generation_params,
            fairness_metrics=fairness_report,
            quality_scores=quality_report,
        )
        files["metadata"] = metadata_path
        
        # 3. Write schema
        if schema:
            schema_path = self.metadata_writer.write_schema(schema, "schema.json")
            files["schema"] = schema_path
        
        # 4. Write data dictionary
        if columns:
            dict_path = self.metadata_writer.write_data_dictionary(
                columns=columns,
                descriptions=column_descriptions,
                sensitive_columns=sensitive_columns,
            )
            files["data_dictionary"] = dict_path
        
        # 5. Create README
        readme_path = self._create_readme(
            data=data,
            generation_params=generation_params,
            fairness_report=fairness_report,
            quality_report=quality_report,
            formats=formats,
            license=license,
        )
        files["readme"] = readme_path
        
        # 6. Create LICENSE file
        license_path = self._create_license(license)
        files["license"] = license_path
        
        return files
    
    def _create_readme(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict],
        generation_params: Optional[Dict],
        fairness_report: Optional[Dict],
        quality_report: Optional[Dict],
        formats: List[str],
        license: str
    ) -> str:
        """Create README file."""
        if isinstance(data, np.ndarray):
            n_samples, n_features = data.shape
        elif isinstance(data, dict):
            first_key = list(data.keys())[0]
            n_samples = len(data[first_key])
            n_features = sum(arr.shape[1] if arr.ndim > 1 else 1 for arr in data.values())
        else:
            n_samples, n_features = data.shape
        
        readme = f"""# {self.package_name}

## Synthetic Data Package

Generated: {datetime.now().isoformat()}

### Overview

This package contains synthetic data generated for research and development purposes.

### Data Description

| Property | Value |
|----------|-------|
| **Samples** | {n_samples:,} |
| **Features** | {n_features} |
| **Formats** | {', '.join(formats)} |

### Files

"""
        
        for fmt in formats:
            readme += f"- `data.{fmt}` - Data in {fmt.upper()} format\n"
        
        readme += """- `metadata.json` - Comprehensive metadata
- `schema.json` - Data schema definition
- `data_dictionary.json` - Column descriptions
- `LICENSE` - License information

### Generation Parameters

```json
"""
        readme += json.dumps(generation_params or {}, indent=2)
        readme += """
```

### Fairness Report

```json
"""
        readme += json.dumps(fairness_report or {}, indent=2, default=str)
        readme += """
```

### Quality Report

```json
"""
        readme += json.dumps(quality_report or {}, indent=2, default=str)
        readme += """
```

### Usage

```python
import pandas as pd

# Load CSV
data = pd.read_csv('data.csv')

# Load Parquet
data = pd.read_parquet('data.parquet')

# Load with metadata
import json
with open('metadata.json') as f:
    metadata = json.load(f)
```

### License

This dataset is released under the {license} license.

### Disclaimer

This is synthetic data generated for research purposes. It does not contain real personal information.
"""
        
        path = self.package_dir / "README.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        return str(path)
    
    def _create_license(self, license: str) -> str:
        """Create LICENSE file."""
        licenses = {
            "CC BY 4.0": """Creative Commons Attribution 4.0 International

Copyright (c) {year}

This work is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
""",
            "MIT": """MIT License

Copyright (c) {year}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
""",
        }
        
        license_text = licenses.get(
            license,
            f"License: {license}\n\nPlease refer to the license documentation for terms of use."
        )
        
        license_text = license_text.format(year=datetime.now().year)
        
        path = self.package_dir / "LICENSE"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(license_text)
        
        return str(path)
    
    def create_manifest(self, files: Dict[str, str]) -> str:
        """
        Create package manifest.
        
        Args:
            files: Dictionary of file types to paths
            
        Returns:
            Manifest file path
        """
        manifest = {
            "package_name": self.package_name,
            "created_at": datetime.now().isoformat(),
            "files": files,
            "checksums": {},
        }
        
        # Compute checksums
        import hashlib
        
        for file_type, path in files.items():
            try:
                with open(path, 'rb') as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                manifest["checksums"][file_type] = checksum
            except Exception:
                pass
        
        manifest_path = self.package_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(manifest_path)
