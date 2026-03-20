"""
API Request Schemas
===================

Pydantic models for validating API request payloads.
Provides comprehensive validation for all endpoints including
generation, evaluation, training, and model management.

Features:
- Type-safe request validation
- Default value handling
- Field constraints and validators
- Multi-modal data support
- Fairness configuration validation
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import json


# ==========================================
# Enums for Type Safety
# ==========================================

class ModalityType(str, Enum):
    """Supported data modalities."""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class FairnessParadigm(str, Enum):
    """Supported fairness paradigms."""
    GROUP = "group"
    INDIVIDUAL = "individual"
    COUNTERFACTUAL = "counterfactual"


class GeneratorType(str, Enum):
    """Supported generator architectures."""
    VAE = "vae"
    BETA_VAE = "beta_vae"
    CVAE = "cvae"
    VAE_GAN = "vae_gan"
    WGAN_GP = "wgan_gp"
    STYLEGAN = "stylegan"
    DDPM = "ddpm"
    DDIM = "ddim"
    LDM = "ldm"


class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    NPZ = "npz"
    PT = "pt"  # PyTorch tensors


class FairnessMetricType(str, Enum):
    """Supported fairness metrics for evaluation."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    DISPARATE_IMPACT = "disparate_impact"
    ACCURACY_PARITY = "accuracy_parity"
    CONSISTENCY = "consistency"
    LIPSCHITZ = "lipschitz"
    COUNTERFACTUAL_INVARIANCE = "counterfactual_invariance"


class TrainingFramework(str, Enum):
    """Supported training frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


# ==========================================
# Base Request Models
# ==========================================

class BaseAPIRequest(BaseModel):
    """Base class for all API requests with common fields."""
    
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        str_strip_whitespace=True,
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided request ID for tracking"
    )
    
    callback_url: Optional[str] = Field(
        default=None,
        description="Optional URL to receive completion callback"
    )


# ==========================================
# Generation Request Schemas
# ==========================================

class FairnessConfigRequest(BaseModel):
    """Configuration for fairness constraints in generation."""
    
    model_config = ConfigDict(extra="forbid")
    
    paradigm: FairnessParadigm = Field(
        default=FairnessParadigm.GROUP,
        description="Fairness paradigm to enforce"
    )
    
    metrics: List[FairnessMetricType] = Field(
        default=[FairnessMetricType.DEMOGRAPHIC_PARITY],
        description="List of fairness metrics to optimize/constrain"
    )
    
    sensitive_attributes: List[str] = Field(
        default_factory=list,
        description="List of sensitive attribute names"
    )
    
    constraint_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum allowed fairness violation"
    )
    
    slack_type: str = Field(
        default="relative",
        description="Type of slack: 'relative' or 'absolute'"
    )
    
    adversary_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for adversarial fairness loss"
    )
    
    lambda_fairness: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Overall fairness regularization weight"
    )
    
    use_adversarial: bool = Field(
        default=True,
        description="Whether to use adversarial debiasing"
    )
    
    use_constrained: bool = Field(
        default=False,
        description="Whether to use constrained optimization"
    )
    
    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, v):
        """Ensure metrics is a list."""
        if isinstance(v, str):
            return [v]
        return v


class GenerationRequest(BaseAPIRequest):
    """
    Request model for synthetic data generation.
    
    This is the primary request type for generating fair synthetic
    data. Supports single and multi-modal generation with configurable
    fairness constraints.
    """
    
    n_samples: int = Field(
        default=1000,
        ge=1,
        le=10000000,
        description="Number of synthetic samples to generate"
    )
    
    modality: ModalityType = Field(
        default=ModalityType.TABULAR,
        description="Data modality for generation"
    )
    
    generator_type: GeneratorType = Field(
        default=GeneratorType.VAE,
        description="Generator architecture to use"
    )
    
    model_name: Optional[str] = Field(
        default=None,
        description="Name of pre-trained model to use (if any)"
    )
    
    fairness_config: Optional[FairnessConfigRequest] = Field(
        default=None,
        description="Fairness constraint configuration"
    )
    
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output data format"
    )
    
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    batch_size: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Batch size for generation"
    )
    
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU acceleration"
    )
    
    precision: str = Field(
        default="float32",
        description="Numerical precision: 'float16', 'float32', 'float64'"
    )
    
    return_latents: bool = Field(
        default=False,
        description="Whether to return latent representations"
    )
    
    postprocess: bool = Field(
        default=True,
        description="Whether to apply post-processing fairness audit"
    )
    
    quality_filter: bool = Field(
        default=True,
        description="Whether to apply quality filtering"
    )
    
    min_quality_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for filtering"
    )
    
    # Schema for tabular data
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data schema with column types and constraints"
    )
    
    # Conditioning for conditional generation
    conditioning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Conditional attributes for guided generation"
    )
    
    # Privacy settings
    apply_dp: bool = Field(
        default=False,
        description="Whether to apply differential privacy"
    )
    
    epsilon: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="DP epsilon parameter"
    )
    
    delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="DP delta parameter"
    )
    
    @model_validator(mode="after")
    def validate_dp_params(self):
        """Validate differential privacy parameters."""
        if self.apply_dp:
            if self.epsilon is None:
                raise ValueError("epsilon must be specified when apply_dp=True")
            if self.epsilon <= 0:
                raise ValueError("epsilon must be positive")
        return self


class MultiModalGenerationRequest(BaseAPIRequest):
    """
    Request model for multi-modal synthetic data generation.
    
    Supports generation across multiple modalities with cross-modal
    consistency constraints.
    """
    
    n_samples: int = Field(
        default=1000,
        ge=1,
        le=10000000,
        description="Number of synthetic samples to generate"
    )
    
    modalities: List[ModalityType] = Field(
        default=[ModalityType.TABULAR],
        min_length=1,
        description="List of modalities to generate"
    )
    
    generator_type: GeneratorType = Field(
        default=GeneratorType.VAE,
        description="Generator architecture for multi-modal fusion"
    )
    
    fairness_config: Optional[FairnessConfigRequest] = Field(
        default=None,
        description="Fairness configuration"
    )
    
    cross_modal_consistency: bool = Field(
        default=True,
        description="Whether to enforce cross-modal consistency"
    )
    
    alignment_method: str = Field(
        default="contrastive",
        description="Method for cross-modal alignment"
    )
    
    output_formats: Dict[str, OutputFormat] = Field(
        default_factory=lambda: {"tabular": OutputFormat.CSV},
        description="Output format per modality"
    )
    
    seed: Optional[int] = Field(default=None)
    
    batch_size: int = Field(default=256, ge=1, le=4096)


class BatchGenerationRequest(BaseAPIRequest):
    """
    Request model for batch generation with multiple configurations.
    
    Allows running multiple generation jobs with different parameters
    in a single request.
    """
    
    configurations: List[GenerationRequest] = Field(
        min_length=1,
        max_length=100,
        description="List of generation configurations"
    )
    
    parallel_jobs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of parallel generation jobs"
    )
    
    combine_outputs: bool = Field(
        default=False,
        description="Whether to combine outputs into single file"
    )
    
    generate_comparison_report: bool = Field(
        default=True,
        description="Whether to generate comparison report"
    )


# ==========================================
# Evaluation Request Schemas
# ==========================================

class EvaluationRequest(BaseAPIRequest):
    """
    Request model for fairness and fidelity evaluation.
    
    Supports comprehensive evaluation of synthetic data including
    fairness metrics, fidelity measures, and privacy assessments.
    """
    
    data_path: str = Field(
        description="Path to synthetic data file"
    )
    
    real_data_path: Optional[str] = Field(
        default=None,
        description="Path to real data for comparison (fidelity metrics)"
    )
    
    evaluation_types: List[str] = Field(
        default=["fairness", "fidelity"],
        description="Types of evaluation to perform"
    )
    
    fairness_metrics: List[FairnessMetricType] = Field(
        default=[FairnessMetricType.DEMOGRAPHIC_PARITY],
        description="Fairness metrics to compute"
    )
    
    fidelity_metrics: List[str] = Field(
        default=["js_divergence", "correlation_preservation", "ks_test"],
        description="Fidelity metrics to compute"
    )
    
    sensitive_attributes: List[str] = Field(
        default_factory=list,
        description="Sensitive attributes for fairness evaluation"
    )
    
    target_column: Optional[str] = Field(
        default=None,
        description="Target column for prediction-based fairness metrics"
    )
    
    # Privacy evaluation settings
    evaluate_privacy: bool = Field(
        default=False,
        description="Whether to perform privacy evaluation"
    )
    
    privacy_attacks: List[str] = Field(
        default=["membership_inference", "attribute_inference"],
        description="Privacy attacks to evaluate"
    )
    
    # Multimodal evaluation settings
    evaluate_multimodal: bool = Field(
        default=False,
        description="Whether to evaluate cross-modal consistency"
    )
    
    # Detailed output settings
    generate_report: bool = Field(
        default=True,
        description="Whether to generate detailed report"
    )
    
    report_format: str = Field(
        default="html",
        description="Report format: 'html', 'json', 'markdown'"
    )
    
    include_visualizations: bool = Field(
        default=True,
        description="Whether to include visualizations in report"
    )
    
    # Threshold validation
    fairness_thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fairness thresholds to validate against"
    )
    
    fidelity_thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fidelity thresholds to validate against"
    )


# ==========================================
# Training Request Schemas
# ==========================================

class TrainingRequest(BaseAPIRequest):
    """
    Request model for training a fair synthetic data generator.
    
    Supports training new models with configurable fairness
    constraints and optimization settings.
    """
    
    training_data_path: str = Field(
        description="Path to training data"
    )
    
    validation_data_path: Optional[str] = Field(
        default=None,
        description="Path to validation data"
    )
    
    model_name: str = Field(
        description="Name for the trained model"
    )
    
    modality: ModalityType = Field(
        default=ModalityType.TABULAR,
        description="Data modality"
    )
    
    generator_type: GeneratorType = Field(
        default=GeneratorType.VAE,
        description="Generator architecture to train"
    )
    
    framework: TrainingFramework = Field(
        default=TrainingFramework.PYTORCH,
        description="Deep learning framework to use"
    )
    
    # Fairness settings
    fairness_config: FairnessConfigRequest = Field(
        default_factory=FairnessConfigRequest,
        description="Fairness training configuration"
    )
    
    # Training hyperparameters
    epochs: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of training epochs"
    )
    
    batch_size: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Training batch size"
    )
    
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-7,
        le=1.0,
        description="Learning rate"
    )
    
    optimizer: str = Field(
        default="adam",
        description="Optimizer: 'adam', 'adamw', 'sgd', 'rmsprop'"
    )
    
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight decay for regularization"
    )
    
    lr_scheduler: str = Field(
        default="cosine",
        description="Learning rate scheduler"
    )
    
    warmup_epochs: int = Field(
        default=10,
        ge=0,
        le=1000,
        description="Number of warmup epochs"
    )
    
    # Architecture parameters
    latent_dim: int = Field(
        default=128,
        ge=8,
        le=4096,
        description="Latent space dimension"
    )
    
    hidden_dims: List[int] = Field(
        default=[256, 512, 256],
        description="Hidden layer dimensions"
    )
    
    # Training strategy
    training_strategy: str = Field(
        default="standard",
        description="Training strategy: 'standard', 'adversarial', 'curriculum', 'multi_task'"
    )
    
    use_amp: bool = Field(
        default=True,
        description="Whether to use automatic mixed precision"
    )
    
    gradient_clip: Optional[float] = Field(
        default=1.0,
        description="Gradient clipping value"
    )
    
    # Checkpointing
    save_checkpoints: bool = Field(
        default=True,
        description="Whether to save training checkpoints"
    )
    
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Epochs between checkpoint saves"
    )
    
    # Early stopping
    early_stopping: bool = Field(
        default=True,
        description="Whether to use early stopping"
    )
    
    patience: int = Field(
        default=10,
        ge=1,
        description="Early stopping patience"
    )
    
    # Distributed training
    distributed: bool = Field(
        default=False,
        description="Whether to use distributed training"
    )
    
    num_gpus: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPUs for distributed training"
    )
    
    # Model architecture config
    architecture_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional architecture configuration"
    )


# ==========================================
# Model Management Request Schemas
# ==========================================

class ModelUploadRequest(BaseAPIRequest):
    """Request model for uploading a pre-trained model."""
    
    model_name: str = Field(
        description="Name for the uploaded model"
    )
    
    model_type: GeneratorType = Field(
        description="Generator architecture type"
    )
    
    modality: ModalityType = Field(
        description="Data modality the model was trained on"
    )
    
    framework: TrainingFramework = Field(
        default=TrainingFramework.PYTORCH,
        description="Framework the model was trained with"
    )
    
    model_path: str = Field(
        description="Path to model weights file"
    )
    
    config_path: Optional[str] = Field(
        default=None,
        description="Path to model configuration file"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Model description"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for model organization"
    )
    
    fairness_type: Optional[str] = Field(
        default=None,
        description="Type of fairness training applied"
    )


class StatusQueryRequest(BaseAPIRequest):
    """Request model for querying job status."""
    
    job_id: str = Field(
        description="Job ID to query"
    )
    
    include_details: bool = Field(
        default=False,
        description="Whether to include detailed progress information"
    )
    
    include_logs: bool = Field(
        default=False,
        description="Whether to include recent log entries"
    )
    
    log_lines: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of log lines to return"
    )
