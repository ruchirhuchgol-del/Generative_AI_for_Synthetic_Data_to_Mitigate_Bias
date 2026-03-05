"""
Image Preprocessor
==================

Preprocessing utilities for image data including:
- Resizing and cropping
- Normalization and standardization
- Color space conversion
- Noise reduction
- Augmentation integration
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)

# Default normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """
    Comprehensive preprocessor for image data.
    
    Handles:
    - Resizing with various interpolation methods
    - Cropping (center, random, corner)
    - Normalization (ImageNet, custom)
    - Color space conversion (RGB, grayscale, HSV)
    - Noise reduction and enhancement
    - Tensor conversion
    
    Attributes:
        target_size: Target image size (height, width)
        normalize: Normalization method
        mean: Normalization mean values
        std: Normalization std values
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        crop_size: Optional[Tuple[int, int]] = None,
        normalize: str = "imagenet",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        color_mode: str = "rgb",
        interpolation: str = "bilinear",
        antialias: bool = True,
        convert_mode: str = "resize",  # resize, crop, pad
        pad_mode: str = "constant",  # constant, reflect, replicate
        pad_value: int = 0,
        enhance_contrast: bool = False,
        denoise: bool = False,
        preserve_aspect_ratio: bool = False
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            crop_size: Crop size if different from target
            normalize: Normalization method ('imagenet', 'custom', 'none')
            mean: Custom mean values for normalization
            std: Custom std values for normalization
            color_mode: Output color mode ('rgb', 'grayscale', 'hsv')
            interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')
            antialias: Use antialiasing during resize
            convert_mode: How to handle size conversion ('resize', 'crop', 'pad')
            pad_mode: Padding mode if convert_mode is 'pad'
            pad_value: Value for constant padding
            enhance_contrast: Apply contrast enhancement
            denoise: Apply denoising
            preserve_aspect_ratio: Preserve aspect ratio during resize
        """
        self.target_size = target_size
        self.crop_size = crop_size or target_size
        self.normalize = normalize
        self.color_mode = color_mode
        self.interpolation = interpolation
        self.antialias = antialias
        self.convert_mode = convert_mode
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Set normalization parameters
        if normalize == "imagenet":
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD
        elif normalize == "custom":
            self.mean = mean or [0.5, 0.5, 0.5]
            self.std = std or [0.5, 0.5, 0.5]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
            
        self._is_fitted = False
        self._statistics: Dict[str, Any] = {}
        
    def fit(self, images: Optional[List[Any]] = None) -> "ImagePreprocessor":
        """
        Fit the preprocessor.
        
        Args:
            images: Optional list of images for computing statistics
            
        Returns:
            Fitted preprocessor
        """
        if images is not None:
            self._compute_statistics(images)
            
        self._is_fitted = True
        logger.info("ImagePreprocessor fitted")
        
        return self
    
    def transform(
        self, 
        image: Any,
        return_type: str = "numpy"
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Transform a single image.
        
        Args:
            image: PIL Image or numpy array
            return_type: Return type ('numpy', 'tensor')
            
        Returns:
            Preprocessed image
        """
        # Convert to PIL Image if needed
        pil_image = self._to_pil(image)
        
        # Convert color mode if needed
        pil_image = self._convert_color_mode(pil_image)
        
        # Resize/crop/pad
        pil_image = self._convert_size(pil_image)
        
        # Convert to numpy
        img_array = np.array(pil_image, dtype=np.float32)
        
        # Apply enhancements
        if self.enhance_contrast:
            img_array = self._enhance_contrast(img_array)
            
        if self.denoise:
            img_array = self._denoise(img_array)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Apply normalization
        img_array = self._normalize(img_array)
        
        # Convert to tensor if requested
        if return_type == "tensor":
            return self._to_tensor(img_array)
        
        return img_array
    
    def transform_batch(
        self,
        images: List[Any],
        return_type: str = "numpy"
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Transform a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            return_type: Return type
            
        Returns:
            Batch of preprocessed images
        """
        processed = [self.transform(img, return_type="numpy") for img in images]
        batch = np.stack(processed)
        
        if return_type == "tensor":
            return self._to_tensor(batch)
        
        return batch
    
    def fit_transform(
        self, 
        images: List[Any],
        return_type: str = "numpy"
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Fit and transform in one step.
        
        Args:
            images: List of images
            return_type: Return type
            
        Returns:
            Preprocessed images
        """
        return self.fit(images).transform_batch(images, return_type)
    
    def inverse_transform(
        self,
        image: Union[np.ndarray, "torch.Tensor"]
    ) -> np.ndarray:
        """
        Inverse transform to original space (denormalize).
        
        Args:
            image: Normalized image array or tensor
            
        Returns:
            Denormalized image array in [0, 255]
        """
        if hasattr(image, 'numpy'):  # torch.Tensor
            image = image.numpy()
            
        # Denormalize
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)
        
        image = image * std + mean
        
        # Clip to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Scale to [0, 255]
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def _to_pil(self, image: Any):
        """Convert various formats to PIL Image."""
        from PIL import Image
        
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _convert_color_mode(self, image) -> Any:
        """Convert image to target color mode."""
        if self.color_mode == "rgb":
            if image.mode != "RGB":
                return image.convert("RGB")
        elif self.color_mode == "grayscale":
            if image.mode != "L":
                return image.convert("L")
        elif self.color_mode == "hsv":
            if image.mode != "HSV":
                return image.convert("HSV")
        return image
    
    def _convert_size(self, image) -> Any:
        """Convert image to target size."""
        if self.convert_mode == "resize":
            return self._resize(image)
        elif self.convert_mode == "crop":
            return self._center_crop(image)
        elif self.convert_mode == "pad":
            return self._pad(image)
        return image
    
    def _resize(self, image) -> Any:
        """Resize image to target size."""
        from PIL import Image
        
        # Get interpolation method
        interp_map = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC
        }
        interpolation = interp_map.get(self.interpolation, Image.Resampling.BILINEAR)
        
        if self.preserve_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            w, h = image.size
            target_h, target_w = self.target_size
            
            ratio = min(target_w / w, target_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            image = image.resize((new_w, new_h), interpolation)
            
            # Pad to target size
            new_image = Image.new(
                image.mode,
                (target_w, target_h),
                self.pad_value
            )
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            new_image.paste(image, (paste_x, paste_y))
            return new_image
        else:
            return image.resize(
                (self.target_size[1], self.target_size[0]),
                interpolation
            )
    
    def _center_crop(self, image) -> Any:
        """Center crop image to target size."""
        w, h = image.size
        target_h, target_w = self.crop_size
        
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        return image.crop((left, top, right, bottom))
    
    def _pad(self, image) -> Any:
        """Pad image to target size."""
        w, h = image.size
        target_h, target_w = self.target_size
        
        pad_left = (target_w - w) // 2
        pad_top = (target_h - h) // 2
        pad_right = target_w - w - pad_left
        pad_bottom = target_h - h - pad_top
        
        from PIL import ImageOps
        return ImageOps.expand(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.pad_value
        )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement."""
        try:
            import cv2
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB).astype(np.float32)
        except ImportError:
            # Simple contrast stretching
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val) * 255
            return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        try:
            import cv2
            return cv2.fastNlMeansDenoisingColored(
                image.astype(np.uint8), 
                None, 
                10, 10, 7, 21
            ).astype(np.float32)
        except ImportError:
            return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using mean and std."""
        mean = np.array(self.mean)
        std = np.array(self.std)
        
        # Reshape for broadcasting
        if image.ndim == 3:
            mean = mean.reshape(1, 1, 3)
            std = std.reshape(1, 1, 3)
            
        return (image - mean) / std
    
    def _to_tensor(self, image: np.ndarray):
        """Convert numpy array to torch tensor."""
        import torch
        
        # Handle different array shapes
        if image.ndim == 3:
            # HWC -> CHW
            tensor = torch.from_numpy(image).permute(2, 0, 1)
        elif image.ndim == 4:
            # BHWC -> BCHW
            tensor = torch.from_numpy(image).permute(0, 3, 1, 2)
        else:
            tensor = torch.from_numpy(image)
            
        return tensor.float()
    
    def _compute_statistics(self, images: List[Any]) -> None:
        """Compute statistics from images."""
        self._statistics = {
            "n_images": len(images),
            "target_size": self.target_size,
            "color_mode": self.color_mode,
            "normalization": self.normalize,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computed statistics."""
        return self._statistics.copy()
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape (C, H, W)."""
        if self.color_mode == "grayscale":
            channels = 1
        else:
            channels = 3
        return (channels, self.target_size[0], self.target_size[1])
    
    def save(self, path: Union[str, Path]) -> None:
        """Save preprocessor configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "target_size": self.target_size,
            "crop_size": self.crop_size,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "color_mode": self.color_mode,
            "interpolation": self.interpolation,
            "antialias": self.antialias,
            "convert_mode": self.convert_mode,
            "pad_mode": self.pad_mode,
            "pad_value": self.pad_value,
            "enhance_contrast": self.enhance_contrast,
            "denoise": self.denoise,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "statistics": self._statistics
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"ImagePreprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ImagePreprocessor":
        """Load preprocessor configuration."""
        with open(path) as f:
            config = json.load(f)
            
        preprocessor = cls(
            target_size=tuple(config["target_size"]),
            crop_size=tuple(config["crop_size"]) if config["crop_size"] else None,
            normalize=config["normalize"],
            mean=config["mean"],
            std=config["std"],
            color_mode=config["color_mode"],
            interpolation=config["interpolation"],
            antialias=config["antialias"],
            convert_mode=config["convert_mode"],
            pad_mode=config["pad_mode"],
            pad_value=config["pad_value"],
            enhance_contrast=config["enhance_contrast"],
            denoise=config["denoise"],
            preserve_aspect_ratio=config["preserve_aspect_ratio"]
        )
        preprocessor._statistics = config.get("statistics", {})
        preprocessor._is_fitted = True
        
        return preprocessor
