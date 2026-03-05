"""
Image Augmenter
===============

Data augmentation utilities for image data including:
- Geometric transformations (flip, rotate, crop, scale)
- Color transformations (brightness, contrast, saturation, hue)
- Noise injection
- Cutout and random erasing
- MixUp and CutMix
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


class ImageAugmenter:
    """
    Data augmenter for image data.
    
    Provides multiple augmentation strategies:
    - Geometric: Flip, rotate, crop, scale, translate
    - Color: Brightness, contrast, saturation, hue
    - Noise: Gaussian noise, blur
    - Occlusion: Cutout, random erasing
    - Mixing: MixUp, CutMix
    
    Attributes:
        augmentation_ratio: Ratio of augmented to original samples
    """
    
    def __init__(
        self,
        augmentation_ratio: float = 0.5,
        # Geometric
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation_range: int = 0,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        zoom_range: float = 0.0,
        # Color
        brightness_range: Optional[Tuple[float, float]] = None,
        contrast_range: Optional[Tuple[float, float]] = None,
        saturation_range: Optional[Tuple[float, float]] = None,
        hue_range: Optional[Tuple[float, float]] = None,
        # Noise
        gaussian_noise_std: float = 0.0,
        gaussian_blur_kernel: int = 0,
        # Occlusion
        cutout_prob: float = 0.0,
        cutout_size: int = 16,
        random_erasing_prob: float = 0.0,
        # Mixing
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        # General
        preserve_aspect_ratio: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the image augmenter.
        
        Args:
            augmentation_ratio: Ratio of augmented samples
            horizontal_flip: Apply horizontal flip
            vertical_flip: Apply vertical flip
            rotation_range: Random rotation range in degrees
            width_shift_range: Random width shift as fraction
            height_shift_range: Random height shift as fraction
            zoom_range: Random zoom range
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            saturation_range: Saturation adjustment range
            hue_range: Hue adjustment range
            gaussian_noise_std: Gaussian noise standard deviation
            gaussian_blur_kernel: Gaussian blur kernel size
            cutout_prob: Cutout probability
            cutout_size: Cutout square size
            random_erasing_prob: Random erasing probability
            mixup_alpha: MixUp alpha parameter
            cutmix_alpha: CutMix alpha parameter
            preserve_aspect_ratio: Preserve aspect ratio
            seed: Random seed
        """
        self.augmentation_ratio = augmentation_ratio
        
        # Geometric
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        
        # Color
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        
        # Noise
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_blur_kernel = gaussian_blur_kernel
        
        # Occlusion
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        self.random_erasing_prob = random_erasing_prob
        
        # Mixing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
        
    def augment(
        self,
        image: np.ndarray,
        return_numpy: bool = True
    ) -> Union[np.ndarray, Any]:
        """
        Augment a single image.
        
        Args:
            image: Input image array (H, W, C)
            return_numpy: Return as numpy array
            
        Returns:
            Augmented image
        """
        # Convert to PIL for easier manipulation
        pil_image = self._to_pil(image)
        
        # Apply geometric transforms
        pil_image = self._apply_geometric(pil_image)
        
        # Apply color transforms
        pil_image = self._apply_color(pil_image)
        
        if return_numpy:
            result = np.array(pil_image, dtype=np.float32)
            
            # Apply noise
            result = self._apply_noise(result)
            
            # Apply occlusion
            result = self._apply_occlusion(result)
            
            return result
            
        return pil_image
    
    def augment_batch(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Augment a batch of images.
        
        Args:
            images: Batch of images (B, H, W, C)
            labels: Optional labels
            
        Returns:
            Augmented images (and labels if mixing is used)
        """
        n_samples = int(len(images) * self.augmentation_ratio)
        
        # Sample indices
        indices = self._rng.choice(len(images), size=n_samples, replace=True)
        
        # Apply mixing augmentations
        if self.mixup_alpha > 0 and labels is not None:
            return self._apply_mixup(images, labels, n_samples)
        elif self.cutmix_alpha > 0 and labels is not None:
            return self._apply_cutmix(images, labels, n_samples)
            
        # Standard augmentation
        augmented = np.stack([
            self.augment(images[i]) for i in indices
        ])
        
        if labels is not None:
            return augmented, labels[indices]
            
        return augmented
    
    def _to_pil(self, image: np.ndarray) -> Any:
        """Convert numpy array to PIL Image."""
        from PIL import Image
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        return Image.fromarray(image)
    
    def _apply_geometric(self, image: Any) -> Any:
        """Apply geometric transformations."""
        from PIL import Image
        
        # Horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
            
        # Translation
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            w, h = image.size
            tx = int(random.uniform(-self.width_shift_range, self.width_shift_range) * w)
            ty = int(random.uniform(-self.height_shift_range, self.height_shift_range) * h)
            image = ImageOps.affine(image, 1, (tx, ty), 0, 0)
            
        # Zoom
        if self.zoom_range > 0:
            zoom = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            w, h = image.size
            new_w, new_h = int(w * zoom), int(h * zoom)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            
            # Crop back to original size
            if zoom > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                image = image.crop((left, top, left + w, top + h))
            else:
                # Pad to original size
                from PIL import ImageOps
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = ImageOps.expand(image, (pad_x, pad_y, w - new_w - pad_x, h - new_h - pad_y))
                
        return image
    
    def _apply_color(self, image: Any) -> Any:
        """Apply color transformations."""
        from PIL import ImageEnhance
        
        # Brightness
        if self.brightness_range is not None:
            factor = random.uniform(*self.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(factor)
            
        # Contrast
        if self.contrast_range is not None:
            factor = random.uniform(*self.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(factor)
            
        # Saturation
        if self.saturation_range is not None:
            factor = random.uniform(*self.saturation_range)
            image = ImageEnhance.Color(image).enhance(factor)
            
        # Hue
        if self.hue_range is not None:
            import colorsys
            factor = random.uniform(*self.hue_range)
            # Convert to HSV, adjust, convert back
            image = image.convert("HSV")
            h, s, v = image.split()
            h = h.point(lambda x: (x + factor * 255) % 256)
            image = Image.merge("HSV", (h, s, v)).convert("RGB")
            
        return image
    
    def _apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise transformations."""
        # Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = self._rng.normal(0, self.gaussian_noise_std, image.shape)
            image = image + noise * 255
            
        # Gaussian blur
        if self.gaussian_blur_kernel > 0:
            try:
                import cv2
                image = cv2.GaussianBlur(
                    image.astype(np.uint8), 
                    (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 
                    0
                )
            except ImportError:
                from PIL import ImageFilter
                pil_img = Image.fromarray(image.astype(np.uint8))
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(self.gaussian_blur_kernel // 2))
                image = np.array(pil_img)
                
        return np.clip(image, 0, 255)
    
    def _apply_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Apply occlusion transformations."""
        h, w = image.shape[:2]
        
        # Cutout
        if self.cutout_prob > 0 and random.random() < self.cutout_prob:
            x = random.randint(0, w - self.cutout_size)
            y = random.randint(0, h - self.cutout_size)
            image[y:y+self.cutout_size, x:x+self.cutout_size] = 0
            
        # Random erasing
        if self.random_erasing_prob > 0 and random.random() < self.random_erasing_prob:
            erase_h = random.randint(h // 8, h // 4)
            erase_w = random.randint(w // 8, w // 4)
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            image[y:y+erase_h, x:x+erase_w] = self._rng.integers(0, 256)
            
        return image
    
    def _apply_mixup(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply MixUp augmentation."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, n_samples)
        
        indices1 = self._rng.choice(len(images), size=n_samples, replace=True)
        indices2 = self._rng.choice(len(images), size=n_samples, replace=True)
        
        lam = lam.reshape(-1, 1, 1, 1)
        mixed_images = lam * images[indices1] + (1 - lam) * images[indices2]
        
        # Soft labels
        lam = lam.reshape(-1)
        mixed_labels = np.zeros((n_samples, labels.max() + 1))
        mixed_labels[np.arange(n_samples), labels[indices1]] = lam
        mixed_labels[np.arange(n_samples), labels[indices2]] += (1 - lam)
        
        return mixed_images, mixed_labels
    
    def _apply_cutmix(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CutMix augmentation."""
        indices1 = self._rng.choice(len(images), size=n_samples, replace=True)
        indices2 = self._rng.choice(len(images), size=n_samples, replace=True)
        
        h, w = images.shape[1:3]
        mixed_images = images[indices1].copy()
        
        # Generate random box
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, n_samples)
        cut_w = (w * np.sqrt(1 - lam)).astype(int)
        cut_h = (h * np.sqrt(1 - lam)).astype(int)
        
        for i in range(n_samples):
            cx = random.randint(0, w - cut_w[i])
            cy = random.randint(0, h - cut_h[i])
            mixed_images[i, cy:cy+cut_h[i], cx:cx+cut_w[i]] = \
                images[indices2[i], cy:cy+cut_h[i], cx:cx+cut_w[i]]
                
        # Soft labels
        mixed_labels = np.zeros((n_samples, labels.max() + 1))
        mixed_labels[np.arange(n_samples), labels[indices1]] = lam
        mixed_labels[np.arange(n_samples), labels[indices2]] += (1 - lam)
        
        return mixed_images, mixed_labels


class AlbumentationsAdapter:
    """
    Adapter for using albumentations library.
    
    Provides a convenient interface for albumentations transforms.
    """
    
    def __init__(self, transform_list: Optional[List[Dict]] = None):
        """
        Initialize with transform configurations.
        
        Args:
            transform_list: List of transform configurations
        """
        try:
            import albumentations as A
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("albumentations not installed. Adapter disabled.")
            return
            
        if transform_list is None:
            # Default transforms
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        else:
            transforms = []
            for t in transform_list:
                name = t.pop("name")
                prob = t.pop("p", 0.5)
                transform_cls = getattr(A, name)
                transforms.append(transform_cls(p=prob, **t))
            self.transform = A.Compose(transforms)
            
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transforms to image."""
        if not self._available:
            return image
        return self.transform(image=image)["image"]
    
    @classmethod
    def from_preset(cls, preset: str = "light") -> "AlbumentationsAdapter":
        """
        Create from preset configuration.
        
        Args:
            preset: Preset name ('light', 'medium', 'heavy')
            
        Returns:
            Configured adapter
        """
        presets = {
            "light": [
                {"name": "HorizontalFlip", "p": 0.5},
                {"name": "RandomBrightnessContrast", "p": 0.2},
            ],
            "medium": [
                {"name": "HorizontalFlip", "p": 0.5},
                {"name": "RandomBrightnessContrast", "p": 0.3},
                {"name": "GaussNoise", "p": 0.2},
                {"name": "Rotate", "limit": 15, "p": 0.3},
            ],
            "heavy": [
                {"name": "HorizontalFlip", "p": 0.5},
                {"name": "RandomBrightnessContrast", "p": 0.5},
                {"name": "GaussNoise", "p": 0.3},
                {"name": "Rotate", "limit": 30, "p": 0.5},
                {"name": "ShiftScaleRotate", "p": 0.3},
                {"name": "CoarseDropout", "p": 0.2},
            ]
        }
        
        return cls(presets.get(preset, presets["light"]))
