"""
Configuration Package
=====================

This package provides configuration management for the Fair Synthetic Data Generator.
Supports YAML-based configuration with hierarchical inheritance and parameter interpolation.
"""

from .config_loader import ConfigLoader, ConfigManager
from .config_validator import ConfigValidator

__all__ = ["ConfigLoader", "ConfigManager", "ConfigValidator"]
