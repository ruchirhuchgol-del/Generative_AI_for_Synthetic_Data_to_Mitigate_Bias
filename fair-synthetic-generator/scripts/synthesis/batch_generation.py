#!/usr/bin/env python
"""
Batch Generation Script
=======================

Script for large-scale batch generation of synthetic data.
Supports parallel processing, distributed generation, and incremental output.

Usage:
    python batch_generation.py [OPTIONS]

Options:
    --checkpoint PATH     Path to model checkpoint
    --output-dir PATH     Output directory for generated data
    --total-samples N     Total number of samples to generate
    --batch-size N        Batch size for generation
    --n-workers N         Number of parallel workers
    --output-format FMT   Output format: csv, parquet, numpy, hdf5
    --split-size N        Split output files by number of samples
    --compress            Compress output files
    --seed N              Random seed
    -h, --help            Show this help message

Examples:
    python batch_generation.py --checkpoint model.pt --total-samples 1000000 --n-workers 4
    python batch_generation.py --checkpoint model.pt --output-format parquet --compress
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Generation of Synthetic Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated data"
    )
    
    parser.add_argument(
        "--total-samples",
        type=int,
        default=100000,
        help="Total number of samples to generate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for generation"
    )
    
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "parquet", "numpy", "hdf5", "tfrecord"],
        default="csv",
        help="Output format for generated data"
    )
    
    parser.add_argument(
        "--split-size",
        type=int,
        default=None,
        help="Split output files by number of samples"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output files"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for generation (cuda, cpu, auto)"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed generation across GPUs"
    )
    
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress bar"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated data"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata file for generation constraints"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming generation for memory efficiency"
    )
    
    return parser.parse_args()


class BatchGenerator:
    """
    Large-scale batch generation manager.
    
    Features:
    - Parallel generation across multiple workers
    - Memory-efficient streaming output
    - Multiple output formats
    - Progress tracking and resumable generation
    - Distributed GPU support
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        total_samples: int,
        batch_size: int = 1024,
        n_workers: int = 1,
        output_format: str = "csv",
        split_size: Optional[int] = None,
        compress: bool = False,
        seed: Optional[int] = None,
        device: str = "auto",
        streaming: bool = False,
        progress: bool = True
    ):
        """
        Initialize the batch generator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_dir: Output directory
            total_samples: Total samples to generate
            batch_size: Batch size for generation
            n_workers: Number of parallel workers
            output_format: Output format
            split_size: Split files by sample count
            compress: Compress output files
            seed: Random seed
            device: Device for generation
            streaming: Use streaming mode
            progress: Show progress
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.output_format = output_format
        self.split_size = split_size or total_samples
        self.compress = compress
        self.seed = seed
        self.device = device
        self.streaming = streaming
        self.progress = progress
        
        self.model = None
        self.model_config = None
        self.generated_count = 0
        self.current_split = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation state
        self.state_file = self.output_dir / "generation_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load generation state for resumable generation."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            self.generated_count = state.get("generated_count", 0)
            self.current_split = state.get("current_split", 0)
            print(f"Resuming from {self.generated_count} samples")
    
    def _save_state(self):
        """Save generation state."""
        state = {
            "generated_count": self.generated_count,
            "current_split": self.current_split,
            "total_samples": self.total_samples,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model_config = checkpoint.get('config', {})
        
        # Determine model type and create model
        from src.models.architectures import DebiasedVAE
        
        latent_dim = self.model_config.get('model', {}).get('latent_dim', 512)
        
        self.model = DebiasedVAE(
            modalities=['tabular'],
            latent_dim=latent_dim,
            num_sensitive_groups=2
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Determine device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def generate_batch(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a single batch of samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for this batch
            
        Returns:
            Generated samples as numpy array
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        with torch.no_grad():
            # Sample from prior
            latent_dim = self.model_config.get('model', {}).get('latent_dim', 512)
            z = torch.randn(n_samples, latent_dim, device=self.device)
            
            # Generate
            output = self.model.decode(z)
            
            # Convert to numpy
            if isinstance(output, dict):
                output = output.get('tabular', output[list(output.keys())[0]])
            
            return output.cpu().numpy()
    
    def generate_stream(
        self,
        total_samples: int,
        batch_size: int
    ) -> Iterator[np.ndarray]:
        """
        Generate samples in a streaming fashion.
        
        Args:
            total_samples: Total samples to generate
            batch_size: Batch size
            
        Yields:
            Batches of generated samples
        """
        remaining = total_samples
        
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            
            # Generate batch
            batch_seed = self.seed + self.generated_count if self.seed else None
            batch = self.generate_batch(current_batch_size, seed=batch_seed)
            
            self.generated_count += current_batch_size
            remaining -= current_batch_size
            
            yield batch
    
    def save_batch(
        self,
        batch: np.ndarray,
        split_index: int,
        batch_index: int
    ) -> str:
        """
        Save a batch of generated data.
        
        Args:
            batch: Batch data
            split_index: Current split index
            batch_index: Batch index within split
            
        Returns:
            Path to saved file
        """
        # Determine filename
        if self.compress:
            filename = f"split_{split_index:04d}_batch_{batch_index:06d}.{self.output_format}.gz"
        else:
            filename = f"split_{split_index:04d}_batch_{batch_index:06d}.{self.output_format}"
        
        filepath = self.output_dir / filename
        
        # Save based on format
        if self.output_format == "csv":
            import pandas as pd
            df = pd.DataFrame(batch)
            if self.compress:
                df.to_csv(filepath, index=False, compression='gzip')
            else:
                df.to_csv(filepath, index=False)
                
        elif self.output_format == "parquet":
            import pandas as pd
            df = pd.DataFrame(batch)
            compression = 'gzip' if self.compress else None
            df.to_parquet(filepath, compression=compression)
            
        elif self.output_format == "numpy":
            if self.compress:
                np.savez_compressed(filepath, data=batch)
            else:
                np.save(filepath, batch)
                
        elif self.output_format == "hdf5":
            import h5py
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=batch, compression='gzip' if self.compress else None)
        
        return str(filepath)
    
    def save_split(
        self,
        data: np.ndarray,
        split_index: int
    ) -> str:
        """
        Save a complete split of generated data.
        
        Args:
            data: Generated data for this split
            split_index: Split index
            
        Returns:
            Path to saved file
        """
        if self.compress:
            filename = f"synthetic_split_{split_index:04d}.{self.output_format}.gz"
        else:
            filename = f"synthetic_split_{split_index:04d}.{self.output_format}"
        
        filepath = self.output_dir / filename
        
        if self.output_format == "csv":
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, compression='gzip' if self.compress else None)
            
        elif self.output_format == "parquet":
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_parquet(filepath, compression='gzip' if self.compress else None)
            
        elif self.output_format == "numpy":
            if self.compress:
                np.savez_compressed(filepath, data=data)
            else:
                np.save(filepath, data)
                
        elif self.output_format == "hdf5":
            import h5py
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=data, compression='gzip' if self.compress else None)
        
        return str(filepath)
    
    def generate_parallel(
        self,
        n_samples: int,
        n_workers: int
    ) -> List[np.ndarray]:
        """
        Generate samples in parallel.
        
        Args:
            n_samples: Total samples to generate
            n_workers: Number of parallel workers
            
        Returns:
            List of generated batches
        """
        samples_per_worker = n_samples // n_workers
        remaining = n_samples % n_workers
        
        def generate_worker(worker_id: int, count: int) -> np.ndarray:
            """Worker function for parallel generation."""
            worker_seed = self.seed + worker_id if self.seed else None
            return self.generate_batch(count, seed=worker_seed)
        
        batches = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for worker_id in range(n_workers):
                count = samples_per_worker + (1 if worker_id < remaining else 0)
                future = executor.submit(generate_worker, worker_id, count)
                futures.append(future)
            
            for future in as_completed(futures):
                batches.append(future.result())
        
        return batches
    
    def run(self):
        """Run the batch generation process."""
        print("\n" + "=" * 60)
        print("Batch Generation Started")
        print("=" * 60)
        print(f"Total samples: {self.total_samples:,}")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Workers: {self.n_workers}")
        print(f"Output format: {self.output_format}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60 + "\n")
        
        # Load model
        self.load_model()
        
        start_time = time.time()
        
        # Generate in streaming mode or batch mode
        if self.streaming:
            self._run_streaming()
        else:
            self._run_batch()
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        self._generate_summary(elapsed_time)
        
        print(f"\nGeneration complete!")
        print(f"  Total samples: {self.generated_count:,}")
        print(f"  Time elapsed: {elapsed_time:.2f}s")
        print(f"  Samples/second: {self.generated_count / elapsed_time:.2f}")
    
    def _run_streaming(self):
        """Run streaming generation."""
        split_data = []
        split_index = self.current_split
        samples_in_split = 0
        
        total_batches = (self.total_samples - self.generated_count + self.batch_size - 1) // self.batch_size
        
        if self.progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_batches, desc="Generating")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        for batch in self.generate_stream(
            self.total_samples - self.generated_count,
            self.batch_size
        ):
            split_data.append(batch)
            samples_in_split += len(batch)
            
            # Save split if full
            if samples_in_split >= self.split_size:
                all_data = np.vstack(split_data)
                self.save_split(all_data, split_index)
                
                split_data = []
                samples_in_split = 0
                split_index += 1
                self.current_split = split_index
                
                # Save state
                self._save_state()
            
            if pbar:
                pbar.update(1)
        
        # Save remaining data
        if split_data:
            all_data = np.vstack(split_data)
            self.save_split(all_data, split_index)
        
        if pbar:
            pbar.close()
    
    def _run_batch(self):
        """Run batch generation."""
        remaining = self.total_samples - self.generated_count
        split_index = self.current_split
        
        while remaining > 0:
            current_split_size = min(self.split_size, remaining)
            
            print(f"\nGenerating split {split_index + 1} ({current_split_size:,} samples)...")
            
            # Generate samples for this split
            if self.n_workers > 1:
                batches = self.generate_parallel(current_split_size, self.n_workers)
                all_data = np.vstack(batches)
            else:
                all_data = np.vstack(list(self.generate_stream(current_split_size, self.batch_size)))
            
            # Save split
            filepath = self.save_split(all_data, split_index)
            print(f"Saved: {filepath}")
            
            remaining -= current_split_size
            split_index += 1
            self.current_split = split_index
            
            # Save state
            self._save_state()
    
    def _generate_summary(self, elapsed_time: float):
        """Generate summary file."""
        summary = {
            "total_samples": self.generated_count,
            "batch_size": self.batch_size,
            "n_workers": self.n_workers,
            "output_format": self.output_format,
            "split_size": self.split_size,
            "compressed": self.compress,
            "seed": self.seed,
            "device": str(self.device),
            "elapsed_time_seconds": elapsed_time,
            "samples_per_second": self.generated_count / elapsed_time if elapsed_time > 0 else 0,
            "checkpoint": str(self.checkpoint_path),
            "output_directory": str(self.output_dir),
            "model_config": self.model_config,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary saved to: {summary_path}")


def validate_generated_data(output_dir: str):
    """
    Validate generated data files.
    
    Args:
        output_dir: Directory containing generated data
    """
    print("\nValidating generated data...")
    
    output_path = Path(output_dir)
    
    # Find all data files
    files = list(output_path.glob("*.csv")) + \
            list(output_path.glob("*.parquet")) + \
            list(output_path.glob("*.npy")) + \
            list(output_path.glob("*.npz"))
    
    if not files:
        print("No data files found for validation")
        return
    
    total_samples = 0
    total_size = 0
    
    for filepath in files:
        if filepath.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(filepath)
            n_samples = len(df)
        elif filepath.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(filepath)
            n_samples = len(df)
        elif filepath.suffix == ".npy":
            data = np.load(filepath)
            n_samples = len(data)
        elif filepath.suffix == ".npz":
            data = np.load(filepath)
            n_samples = len(data['data'])
        else:
            continue
        
        file_size = filepath.stat().st_size
        total_samples += n_samples
        total_size += file_size
        
        print(f"  {filepath.name}: {n_samples:,} samples, {file_size / 1024 / 1024:.2f} MB")
    
    print(f"\nTotal: {total_samples:,} samples, {total_size / 1024 / 1024 / 1024:.2f} GB")


def main():
    """Main function."""
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "synthetic" / "batch"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create batch generator
    generator = BatchGenerator(
        checkpoint_path=args.checkpoint,
        output_dir=str(output_dir),
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        output_format=args.output_format,
        split_size=args.split_size,
        compress=args.compress,
        seed=args.seed,
        device=args.device,
        streaming=args.streaming,
        progress=args.progress
    )
    
    # Run generation
    generator.run()
    
    # Validate if requested
    if args.validate:
        validate_generated_data(str(output_dir))


if __name__ == "__main__":
    main()
