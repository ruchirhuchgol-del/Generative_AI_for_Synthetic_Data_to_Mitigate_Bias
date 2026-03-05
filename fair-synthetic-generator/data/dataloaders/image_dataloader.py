"""
Image Dataloader
================

Data loader for image data with preprocessing and augmentation.
Supports various image formats and fairness-aware image processing.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import logging
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from .base_dataloader import BaseDataset, BaseDataLoader

logger = logging.getLogger(__name__)

# Default ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageTransform:
    """
    Image transformation pipeline.
    
    Supports common transforms:
    - Resize
    - Center crop
    - Random crop
    - Random flip
    - Color jitter
    - Normalization
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        crop_size: Optional[Tuple[int, int]] = None,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
        augmentation: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Initialize image transform.
        
        Args:
            size: Target image size (height, width)
            crop_size: Center/random crop size
            mean: Normalization mean
            std: Normalization std
            augmentation: Enable augmentation
            augmentation_config: Augmentation parameters
        """
        self.size = size
        self.crop_size = crop_size or size
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        self.augmentation = augmentation
        self.aug_config = augmentation_config or {}
        
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply transforms to image.
        
        Args:
            image: PIL Image
            
        Returns:
            Transformed tensor
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize
        image = image.resize((self.size[1], self.size[0]), Image.BILINEAR)
        
        # Augmentation
        if self.augmentation:
            image = self._apply_augmentation(image)
            
        # Convert to numpy and normalize
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
        return tensor
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply augmentation transforms."""
        import random
        
        # Random horizontal flip
        if self.aug_config.get("random_flip", {}).get("enabled", False):
            prob = self.aug_config["random_flip"].get("probability", 0.5)
            if random.random() < prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
        # Random rotation
        if self.aug_config.get("random_rotation", {}).get("enabled", False):
            max_angle = self.aug_config["random_rotation"].get("max_angle", 15)
            angle = random.uniform(-max_angle, max_angle)
            image = image.rotate(angle)
            
        # Color jitter
        if self.aug_config.get("color_jitter", {}).get("enabled", False):
            params = self.aug_config["color_jitter"]
            brightness = params.get("brightness", 0.1)
            contrast = params.get("contrast", 0.1)
            saturation = params.get("saturation", 0.1)
            
            # Apply brightness
            from PIL import ImageEnhance
            if brightness > 0:
                factor = 1 + random.uniform(-brightness, brightness)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if contrast > 0:
                factor = 1 + random.uniform(-contrast, contrast)
                image = ImageEnhance.Contrast(image).enhance(factor)
            if saturation > 0:
                factor = 1 + random.uniform(-saturation, saturation)
                image = ImageEnhance.Color(image).enhance(factor)
                
        return image


class ImageDataset(BaseDataset):
    """
    Dataset for image data.
    
    Features:
    - Multiple image format support
    - Automatic preprocessing
    - Augmentation support
    - Sensitive content handling
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        image_paths: Optional[List[Union[str, Path]]] = None,
        labels: Optional[List[Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        transform: Optional[Union[ImageTransform, Callable]] = None,
        size: Tuple[int, int] = (256, 256),
        channels: int = 3,
        label_column: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        load_in_memory: bool = False
    ):
        """
        Initialize the image dataset.
        
        Args:
            data_path: Path to image directory or metadata file
            image_paths: Direct list of image paths
            labels: Optional labels for each image
            schema_path: Path to schema file
            transform: Image transform
            size: Image size (height, width)
            channels: Number of channels
            label_column: Label column name in metadata
            sensitive_attributes: Sensitive attribute names
            load_in_memory: Load all images into memory
        """
        super().__init__(data_path, transform, sensitive_attributes)
        
        self.image_paths: List[Path] = [
            Path(p) for p in image_paths
        ] if image_paths else []
        self.labels: List[Any] = labels or []
        self.schema_path = Path(schema_path) if schema_path else None
        self.size = size
        self.channels = channels
        self.label_column = label_column
        self.load_in_memory = load_in_memory
        
        self._images: Dict[int, Image.Image] = {}
        self._normalization = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
        
        # Create default transform if not provided
        if transform is None:
            self.transform = ImageTransform(size=size)
            
        # Load schema if provided
        if self.schema_path:
            self._load_schema()
            
        # Load data if path provided
        if self.data_path and not self.image_paths:
            self.load_data()
            
    def _load_schema(self) -> None:
        """Load schema from JSON file."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}")
            return
            
        with open(self.schema_path) as f:
            schema = json.load(f)
            
        if schema.get("fields"):
            field = schema["fields"][0]
            dims = field.get("dimensions", {})
            self.size = (dims.get("height", 256), dims.get("width", 256))
            self.channels = dims.get("channels", 3)
            
            if "normalization" in field:
                norm = field["normalization"]
                self._normalization = {
                    "mean": norm.get("mean", IMAGENET_MEAN),
                    "std": norm.get("std", IMAGENET_STD)
                }
                
        self._metadata["schema"] = schema
        
    def load_data(self) -> None:
        """Load data from directory or metadata file."""
        if not self.data_path:
            raise ValueError("No data path specified")
            
        if self.data_path.is_dir():
            # Load from directory
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
            self.image_paths = sorted([
                p for p in self.data_path.rglob("*")
                if p.suffix.lower() in valid_extensions
            ])
            
        elif self.data_path.suffix in [".csv", ".json"]:
            # Load from metadata file
            self._load_from_metadata()
            
        else:
            raise ValueError(f"Unsupported data path type: {self.data_path}")
            
        # Load images into memory if requested
        if self.load_in_memory:
            self._load_images_in_memory()
            
        logger.info(f"Loaded {len(self.image_paths)} image paths")
        
    def _load_from_metadata(self) -> None:
        """Load image paths from metadata file."""
        if self.data_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(self.data_path)
            
            # Find image path column
            path_col = None
            for col in ["image_path", "path", "filepath", "file"]:
                if col in df.columns:
                    path_col = col
                    break
                    
            if path_col:
                base_dir = self.data_path.parent
                self.image_paths = [
                    base_dir / p if not Path(p).is_absolute() else Path(p)
                    for p in df[path_col]
                ]
                
            if self.label_column and self.label_column in df.columns:
                self.labels = df[self.label_column].tolist()
                
        elif self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if "path" in item or "image_path" in item:
                        self.image_paths.append(
                            Path(item.get("path") or item.get("image_path"))
                        )
                    if self.label_column:
                        self.labels.append(item.get(self.label_column))
                        
    def _load_images_in_memory(self) -> None:
        """Load all images into memory."""
        for i, path in enumerate(self.image_paths):
            try:
                self._images[i] = Image.open(path)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_paths)
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with image tensor and optional label
        """
        # Load image
        if index in self._images:
            image = self._images[index].copy()
        else:
            try:
                image = Image.open(self.image_paths[index])
            except Exception as e:
                logger.error(f"Failed to load image {self.image_paths[index]}: {e}")
                # Return blank image
                image = Image.new("RGB", self.size, (0, 0, 0))
                
        # Apply transform
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default transform
            image = image.resize((self.size[1], self.size[0]))
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            
        sample = {
            "image": image_tensor,
            "path": str(self.image_paths[index]),
            "index": index
        }
        
        # Add label if present
        if self.labels and index < len(self.labels):
            label = self.labels[index]
            if isinstance(label, str):
                label = hash(label) % 10000
            sample["label"] = torch.tensor(label, dtype=torch.long)
            
        return sample
    
    def get_image_size(self) -> Tuple[int, int, int]:
        """Get image dimensions (C, H, W)."""
        return (self.channels, self.size[0], self.size[1])
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize image tensor.
        
        Args:
            tensor: Normalized image tensor [C, H, W]
            
        Returns:
            Denormalized tensor in [0, 1] range
        """
        mean = torch.tensor(self._normalization["mean"]).view(-1, 1, 1)
        std = torch.tensor(self._normalization["std"]).view(-1, 1, 1)
        return tensor * std + mean
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_statistics()
        stats.update({
            "num_images": len(self.image_paths),
            "image_size": self.get_image_size(),
            "channels": self.channels,
        })
        return stats


class ImageDataLoader(BaseDataLoader):
    """
    Data loader for image data.
    """
    
    def __init__(
        self,
        dataset: ImageDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler_strategy: Optional[str] = None
    ):
        """
        Initialize the image data loader.
        
        Args:
            dataset: Image dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            drop_last: Drop last batch
            sampler_strategy: Fairness sampling strategy
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler_strategy=sampler_strategy,
            group_labels=None
        )
        
    def show_batch(self, n: int = 8) -> None:
        """
        Display a batch of images (requires matplotlib).
        
        Args:
            n: Number of images to display
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot display images")
            return
            
        batch = next(iter(self._dataloader))
        images = batch["image"][:n]
        
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i, img in enumerate(images):
            img_np = self.dataset.denormalize(img).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[i].imshow(img_np)
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()
