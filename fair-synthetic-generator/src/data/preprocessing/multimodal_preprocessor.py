"""
Multimodal Preprocessor
=======================

Unified preprocessor for multimodal data combining tabular, text, and image modalities.
Provides coordinated preprocessing across modalities with support for:
- Cross-modal alignment
- Unified interface
- Combined preprocessing pipeline
- Synchronized transformations
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .tabular_preprocessor import TabularPreprocessor
from .text_preprocessor import TextPreprocessor
from .image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class MultimodalPreprocessor:
    """
    Unified preprocessor for multimodal data.
    
    Handles preprocessing for:
    - Tabular data: normalization, encoding, imputation
    - Text data: cleaning, tokenization, normalization
    - Image data: resizing, normalization, enhancement
    
    Provides coordinated preprocessing across modalities with support for
    cross-modal alignment and synchronized transformations.
    
    Attributes:
        tabular_preprocessor: Preprocessor for tabular data
        text_preprocessor: Preprocessor for text data
        image_preprocessor: Preprocessor for image data
    """
    
    def __init__(
        self,
        tabular_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        align_on: str = "index",
        process_modalities: Optional[List[str]] = None
    ):
        """
        Initialize the multimodal preprocessor.
        
        Args:
            tabular_config: Configuration for tabular preprocessor
            text_config: Configuration for text preprocessor
            image_config: Configuration for image preprocessor
            sensitive_attributes: List of sensitive attribute names
            align_on: How to align modalities ('index', 'id')
            process_modalities: Which modalities to process (None for all available)
        """
        self.tabular_config = tabular_config or {}
        self.text_config = text_config or {}
        self.image_config = image_config or {}
        self.sensitive_attributes = sensitive_attributes or []
        self.align_on = align_on
        self.process_modalities = process_modalities
        
        # Initialize sub-preprocessors
        self.tabular_preprocessor: Optional[TabularPreprocessor] = None
        self.text_preprocessor: Optional[TextPreprocessor] = None
        self.image_preprocessor: Optional[ImagePreprocessor] = None
        
        # Metadata
        self._modalities: List[str] = []
        self._is_fitted = False
        self._statistics: Dict[str, Any] = {}
        
    @property
    def modalities(self) -> List[str]:
        """Get list of available modalities."""
        return self._modalities
    
    def fit(
        self,
        tabular_data: Optional[pd.DataFrame] = None,
        text_data: Optional[List[str]] = None,
        image_data: Optional[List[Any]] = None
    ) -> "MultimodalPreprocessor":
        """
        Fit all modality preprocessors.
        
        Args:
            tabular_data: Tabular DataFrame
            text_data: List of text strings
            image_data: List of images
            
        Returns:
            Fitted preprocessor
        """
        self._modalities = []
        
        # Fit tabular preprocessor
        if tabular_data is not None:
            self._modalities.append("tabular")
            config = {**self.tabular_config}
            if self.sensitive_attributes:
                config["sensitive_attributes"] = self.sensitive_attributes
            self.tabular_preprocessor = TabularPreprocessor(**config)
            self.tabular_preprocessor.fit(tabular_data)
            logger.info("Tabular preprocessor fitted")
            
        # Fit text preprocessor
        if text_data is not None:
            self._modalities.append("text")
            self.text_preprocessor = TextPreprocessor(**self.text_config)
            self.text_preprocessor.fit(text_data)
            logger.info("Text preprocessor fitted")
            
        # Fit image preprocessor
        if image_data is not None:
            self._modalities.append("image")
            self.image_preprocessor = ImagePreprocessor(**self.image_config)
            self.image_preprocessor.fit(image_data)
            logger.info("Image preprocessor fitted")
            
        # Apply process_modalities filter
        if self.process_modalities:
            self._modalities = [m for m in self._modalities if m in self.process_modalities]
            
        # Compute combined statistics
        self._compute_statistics(tabular_data, text_data, image_data)
        
        self._is_fitted = True
        logger.info(f"MultimodalPreprocessor fitted for modalities: {self._modalities}")
        
        return self
    
    def transform(
        self,
        tabular_data: Optional[pd.DataFrame] = None,
        text_data: Optional[List[str]] = None,
        image_data: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Transform all modalities.
        
        Args:
            tabular_data: Tabular DataFrame
            text_data: List of text strings
            image_data: List of images
            
        Returns:
            Dictionary with transformed data for each modality
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        result = {}
        
        # Transform tabular
        if "tabular" in self._modalities and tabular_data is not None:
            if self.tabular_preprocessor is not None:
                result["tabular"] = self.tabular_preprocessor.transform(tabular_data)
                
        # Transform text
        if "text" in self._modalities and text_data is not None:
            if self.text_preprocessor is not None:
                result["text"] = self.text_preprocessor.transform(text_data)
                
        # Transform images
        if "image" in self._modalities and image_data is not None:
            if self.image_preprocessor is not None:
                result["image"] = self.image_preprocessor.transform_batch(image_data)
                
        return result
    
    def fit_transform(
        self,
        tabular_data: Optional[pd.DataFrame] = None,
        text_data: Optional[List[str]] = None,
        image_data: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Fit and transform in one step.
        
        Args:
            tabular_data: Tabular DataFrame
            text_data: List of text strings
            image_data: List of images
            
        Returns:
            Dictionary with transformed data
        """
        return self.fit(tabular_data, text_data, image_data).transform(
            tabular_data, text_data, image_data
        )
    
    def transform_sample(
        self,
        tabular: Optional[pd.Series] = None,
        text: Optional[str] = None,
        image: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Transform a single multimodal sample.
        
        Args:
            tabular: Single row of tabular data
            text: Single text string
            image: Single image
            
        Returns:
            Dictionary with transformed sample
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        result = {}
        
        # Transform tabular
        if tabular is not None and self.tabular_preprocessor is not None:
            df = pd.DataFrame([tabular])
            result["tabular"] = self.tabular_preprocessor.transform(df).iloc[0]
            
        # Transform text
        if text is not None and self.text_preprocessor is not None:
            result["text"] = self.text_preprocessor.transform(text)
            
        # Transform image
        if image is not None and self.image_preprocessor is not None:
            result["image"] = self.image_preprocessor.transform(image)
            
        return result
    
    def inverse_transform(
        self,
        tabular: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Inverse transform to original space.
        
        Args:
            tabular: Transformed tabular data
            image: Transformed image data
            
        Returns:
            Dictionary with inverse-transformed data
        """
        result = {}
        
        if tabular is not None and self.tabular_preprocessor is not None:
            result["tabular"] = self.tabular_preprocessor.inverse_transform(tabular)
            
        if image is not None and self.image_preprocessor is not None:
            if image.ndim == 4:
                result["image"] = [
                    self.image_preprocessor.inverse_transform(img) 
                    for img in image
                ]
            else:
                result["image"] = self.image_preprocessor.inverse_transform(image)
                
        return result
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get feature information for each modality.
        
        Returns:
            Dictionary with feature info per modality
        """
        info = {}
        
        if self.tabular_preprocessor is not None:
            info["tabular"] = {
                "feature_names": self.tabular_preprocessor.get_feature_names(),
                "statistics": self.tabular_preprocessor.get_statistics()
            }
            
        if self.text_preprocessor is not None:
            info["text"] = {
                "statistics": self.text_preprocessor.get_statistics()
            }
            
        if self.image_preprocessor is not None:
            info["image"] = {
                "output_shape": self.image_preprocessor.get_output_shape(),
                "statistics": self.image_preprocessor.get_statistics()
            }
            
        return info
    
    def _compute_statistics(
        self,
        tabular_data: Optional[pd.DataFrame],
        text_data: Optional[List[str]],
        image_data: Optional[List[Any]]
    ) -> None:
        """Compute combined statistics."""
        self._statistics = {
            "modalities": self._modalities,
            "n_samples": None,
            "tabular": {},
            "text": {},
            "image": {}
        }
        
        # Determine sample count
        n_samples = []
        if tabular_data is not None:
            n_samples.append(len(tabular_data))
        if text_data is not None:
            n_samples.append(len(text_data))
        if image_data is not None:
            n_samples.append(len(image_data))
            
        if n_samples:
            self._statistics["n_samples"] = min(n_samples) if len(set(n_samples)) > 1 else n_samples[0]
            
        # Collect modality-specific statistics
        if self.tabular_preprocessor is not None:
            self._statistics["tabular"] = self.tabular_preprocessor.get_statistics()
            
        if self.text_preprocessor is not None:
            self._statistics["text"] = self.text_preprocessor.get_statistics()
            
        if self.image_preprocessor is not None:
            self._statistics["image"] = self.image_preprocessor.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return self._statistics.copy()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save all preprocessor configurations."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save main config
        main_config = {
            "sensitive_attributes": self.sensitive_attributes,
            "align_on": self.align_on,
            "process_modalities": self.process_modalities,
            "modalities": self._modalities,
            "statistics": self._statistics
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(main_config, f, indent=2)
            
        # Save sub-preprocessors
        if self.tabular_preprocessor is not None:
            self.tabular_preprocessor.save(path / "tabular_preprocessor")
            
        if self.text_preprocessor is not None:
            self.text_preprocessor.save(path / "text_preprocessor.json")
            
        if self.image_preprocessor is not None:
            self.image_preprocessor.save(path / "image_preprocessor.json")
            
        logger.info(f"MultimodalPreprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultimodalPreprocessor":
        """Load preprocessor configuration."""
        path = Path(path)
        
        with open(path / "config.json") as f:
            main_config = json.load(f)
            
        preprocessor = cls(
            sensitive_attributes=main_config["sensitive_attributes"],
            align_on=main_config["align_on"],
            process_modalities=main_config["process_modalities"]
        )
        preprocessor._modalities = main_config["modalities"]
        preprocessor._statistics = main_config["statistics"]
        
        # Load sub-preprocessors
        tabular_path = path / "tabular_preprocessor.json"
        if tabular_path.exists():
            preprocessor.tabular_preprocessor = TabularPreprocessor.load(tabular_path)
            
        text_path = path / "text_preprocessor.json"
        if text_path.exists():
            preprocessor.text_preprocessor = TextPreprocessor.load(text_path)
            
        image_path = path / "image_preprocessor.json"
        if image_path.exists():
            preprocessor.image_preprocessor = ImagePreprocessor.load(image_path)
            
        preprocessor._is_fitted = True
        
        return preprocessor


def create_preprocessor(
    modality: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[TabularPreprocessor, TextPreprocessor, ImagePreprocessor]:
    """
    Factory function to create a preprocessor.
    
    Args:
        modality: Modality type ('tabular', 'text', 'image')
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Preprocessor instance
    """
    config = config or {}
    config.update(kwargs)
    
    if modality == "tabular":
        return TabularPreprocessor(**config)
    elif modality == "text":
        return TextPreprocessor(**config)
    elif modality == "image":
        return ImagePreprocessor(**config)
    else:
        raise ValueError(f"Unknown modality: {modality}")
