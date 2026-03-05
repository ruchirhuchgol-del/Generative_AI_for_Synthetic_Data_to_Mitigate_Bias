"""
Configuration Loader
====================

Utilities for loading, merging, and managing YAML configurations.
Supports hierarchical configs with inheritance and variable interpolation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import copy
import re


class ConfigLoader:
    """
    Configuration loader with support for:
    - YAML file loading
    - Hierarchical config inheritance
    - Variable interpolation (${variable})
    - Environment variable substitution
    - Config validation
    """
    
    DEFAULT_CONFIG_DIR = Path(__file__).parent / "default"
    EXPERIMENT_CONFIG_DIR = Path(__file__).parent / "experiments"
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        allow_env_override: bool = True
    ):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Base directory for config files
            allow_env_override: Allow environment variable overrides
        """
        self.config_dir = config_dir or Path(__file__).parent
        self.allow_env_override = allow_env_override
        self._config_cache: Dict[str, Dict] = {}
        
    def load(
        self,
        config_name: str,
        config_type: str = "experiment",
        inherit_defaults: bool = True
    ) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config (without .yaml extension)
            config_type: Type of config ('default', 'experiment', or 'auto')
            inherit_defaults: Whether to inherit from default configs
            
        Returns:
            Merged configuration dictionary
        """
        # Determine config path
        if config_type == "default":
            config_path = self.DEFAULT_CONFIG_DIR / f"{config_name}.yaml"
        elif config_type == "experiment":
            config_path = self.EXPERIMENT_CONFIG_DIR / f"{config_name}.yaml"
        else:
            # Auto-detect
            config_path = self._find_config(config_name)
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load the config
        config = self._load_yaml(config_path)
        
        # Handle inheritance
        if inherit_defaults and "defaults" in config:
            config = self._resolve_inheritance(config)
            
        # Resolve variable interpolation
        config = self._resolve_variables(config)
        
        # Apply environment overrides
        if self.allow_env_override:
            config = self._apply_env_overrides(config)
            
        return config
    
    def load_all_defaults(self) -> Dict[str, Dict]:
        """
        Load all default configurations.
        
        Returns:
            Dictionary of all default configs
        """
        defaults = {}
        
        for config_file in self.DEFAULT_CONFIG_DIR.glob("*.yaml"):
            config_name = config_file.stem
            defaults[config_name] = self._load_yaml(config_file)
            
        return defaults
    
    def merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    
    def _find_config(self, config_name: str) -> Path:
        """Find a config file by name."""
        # Check default directory
        default_path = self.DEFAULT_CONFIG_DIR / f"{config_name}.yaml"
        if default_path.exists():
            return default_path
            
        # Check experiment directory
        experiment_path = self.EXPERIMENT_CONFIG_DIR / f"{config_name}.yaml"
        if experiment_path.exists():
            return experiment_path
            
        raise FileNotFoundError(f"Config '{config_name}' not found")
    
    def _resolve_inheritance(self, config: Dict) -> Dict:
        """Resolve config inheritance from defaults list."""
        defaults = config.pop("defaults", [])
        merged = {}
        
        for default_ref in defaults:
            if isinstance(default_ref, str):
                # Simple reference: "/config_name"
                if default_ref.startswith("/"):
                    config_name = default_ref[1:]
                    default_config = self.load(
                        config_name,
                        config_type="default",
                        inherit_defaults=False
                    )
                    merged = self.merge_configs(merged, default_config)
            elif isinstance(default_ref, dict):
                # Complex reference with overrides
                for key, value in default_ref.items():
                    if key.startswith("/"):
                        config_name = key[1:]
                        default_config = self.load(
                            config_name,
                            config_type="default",
                            inherit_defaults=False
                        )
                        merged = self.merge_configs(merged, default_config)
                        
        # Merge the current config on top
        merged = self.merge_configs(merged, config)
        
        return merged
    
    def _resolve_variables(self, config: Dict, context: Optional[Dict] = None) -> Dict:
        """
        Resolve variable interpolation in config.
        Supports ${variable} syntax.
        """
        if context is None:
            context = config
            
        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Match ${variable} or ${path.to.variable}
                pattern = r'\$\{([^}]+)\}'
                
                def replace_var(match):
                    var_path = match.group(1)
                    resolved = self._get_nested_value(context, var_path)
                    if resolved is None:
                        # Try to resolve from root config
                        resolved = self._get_nested_value(config, var_path)
                    if resolved is None:
                        # Keep original if not found
                        return match.group(0)
                    return str(resolved)
                    
                return re.sub(pattern, replace_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
            
        return resolve_value(config)
    
    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """Get a nested value from config using dot notation."""
        keys = path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
                
        return value
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides."""
        prefix = "FAIR_GEN_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Convert double underscore to nested path
                path = config_key.split("__")
                self._set_nested_value(config, path, value)
                
        return config
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any) -> None:
        """Set a nested value in config using path list."""
        current = config
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # Try to parse the value
        current[path[-1]] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
            
        # None
        if value.lower() in ("none", "null"):
            return None
            
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Float
        try:
            return float(value)
        except ValueError:
            pass
            
        # List (comma-separated)
        if "," in value:
            return [self._parse_env_value(v.strip()) for v in value.split(",")]
            
        return value


class ConfigManager:
    """
    Centralized configuration manager for the entire project.
    Provides a singleton-like interface for accessing configurations.
    """
    
    _instance: Optional["ConfigManager"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the config manager.
        
        Args:
            config_dir: Base directory for configs
            experiment_name: Name of the experiment to load
        """
        if self._initialized:
            return
            
        self.loader = ConfigLoader(config_dir)
        self._configs: Dict[str, Dict] = {}
        self._experiment_name = experiment_name
        self._initialized = True
        
    def load_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load an experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Merged experiment configuration
        """
        if experiment_name not in self._configs:
            self._configs[experiment_name] = self.loader.load(
                experiment_name,
                config_type="experiment"
            )
        return self._configs[experiment_name]
    
    def get_config(
        self,
        key: str,
        default: Any = None,
        experiment_name: Optional[str] = None
    ) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if not found
            experiment_name: Specific experiment config to use
            
        Returns:
            Configuration value
        """
        if experiment_name:
            config = self.load_experiment(experiment_name)
        else:
            config = self._configs.get(self._experiment_name, {})
            
        return self.loader._get_nested_value(config, key) or default
    
    def set_experiment(self, experiment_name: str) -> None:
        """Set the current experiment."""
        self._experiment_name = experiment_name
        self.load_experiment(experiment_name)
        
    @property
    def current_config(self) -> Dict[str, Any]:
        """Get the current experiment configuration."""
        if self._experiment_name is None:
            raise ValueError("No experiment set")
        return self._configs[self._experiment_name]
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        
    def save_config(
        self,
        config: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save a configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save the config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Convenience function
def load_config(
    config_name: str,
    config_type: str = "experiment"
) -> Dict[str, Any]:
    """
    Convenience function to load a configuration.
    
    Args:
        config_name: Name of the configuration
        config_type: Type of configuration
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load(config_name, config_type=config_type)
