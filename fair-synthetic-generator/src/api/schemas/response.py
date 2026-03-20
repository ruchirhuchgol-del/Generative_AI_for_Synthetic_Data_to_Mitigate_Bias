"""
API Response Schemas
====================

Pydantic models for API response payloads.
Provides consistent response structures across all endpoints.

Features:
- Type-safe response serialization
- Detailed error handling
- Progress tracking
- Comprehensive metadata
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ==========================================
# Enums for Response Status
# ==========================================

class JobStatus(str, Enum):
    """Status of async jobs."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ==========================================
# Base Response Models
# ==========================================

class BaseAPIResponse(BaseModel):
    """Base class for all API responses."""
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for flexibility
        use_enum_values=True,
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class ErrorResponse(BaseAPIResponse):
    """
    Standard error response structure.
    
    Provides detailed error information including
    error codes, messages, and debugging hints.
    """
    
    success: Literal[False] = False
    
    error_code: str = Field(
        description="Machine-readable error code"
    )
    
    error_message: str = Field(
        description="Human-readable error message"
    )
    
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details for debugging"
    )
    
    suggestion: Optional[str] = Field(
        default=None,
        description="Suggested action to resolve the error"
    )
    
    documentation_url: Optional[str] = Field(
        default=None,
        description="Link to relevant documentation"
    )
    
    @classmethod
    def create(
        cls,
        error_code: str,
        message: str,
        details: Optional[Dict] = None,
        suggestion: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> "ErrorResponse":
        """Factory method to create error responses."""
        return cls(
            request_id=request_id,
            error_code=error_code,
            error_message=message,
            error_details=details,
            suggestion=suggestion
        )


# ==========================================
# Generation Response Schemas
# ==========================================

class GenerationResponse(BaseAPIResponse):
    """
    Response model for synthetic data generation request.
    
    Returns job information for async generation tasks.
    """
    
    success: Literal[True] = True
    
    job_id: str = Field(
        description="Unique job identifier"
    )
    
    status: JobStatus = Field(
        description="Current job status"
    )
    
    n_samples: int = Field(
        description="Number of samples requested"
    )
    
    modality: str = Field(
        description="Data modality"
    )
    
    message: str = Field(
        description="Status message"
    )
    
    estimated_time: Optional[float] = Field(
        default=None,
        description="Estimated completion time in seconds"
    )
    
    queue_position: Optional[int] = Field(
        default=None,
        description="Position in job queue"
    )
    
    status_url: str = Field(
        description="URL to check job status"
    )


class GenerationResult(BaseAPIResponse):
    """
    Detailed result of a completed generation job.
    
    Contains metadata about generated data and download URLs.
    """
    
    success: Literal[True] = True
    
    job_id: str
    
    n_samples_generated: int = Field(
        description="Actual number of samples generated"
    )
    
    generation_time: float = Field(
        description="Time taken for generation in seconds"
    )
    
    data_url: str = Field(
        description="Download URL for generated data"
    )
    
    data_format: str = Field(
        description="Format of generated data"
    )
    
    data_size_bytes: int = Field(
        description="Size of generated data in bytes"
    )
    
    expires_at: datetime = Field(
        description="Expiration time for download URL"
    )
    
    # Quality metrics
    quality_score: Optional[float] = Field(
        default=None,
        description="Overall quality score (0-1)"
    )
    
    fidelity_score: Optional[float] = Field(
        default=None,
        description="Fidelity score if real data was provided"
    )
    
    fairness_score: Optional[float] = Field(
        default=None,
        description="Fairness compliance score"
    )
    
    # Additional artifacts
    report_url: Optional[str] = Field(
        default=None,
        description="URL to fairness/quality report"
    )
    
    latent_url: Optional[str] = Field(
        default=None,
        description="URL to latent representations if requested"
    )
    
    # Metadata
    generator_model: str = Field(
        description="Generator model used"
    )
    
    seed_used: Optional[int] = Field(
        default=None,
        description="Random seed used for generation"
    )
    
    checksum: str = Field(
        description="SHA256 checksum for data integrity"
    )


class BatchGenerationResponse(BaseAPIResponse):
    """Response for batch generation requests."""
    
    success: Literal[True] = True
    
    batch_id: str = Field(
        description="Batch job identifier"
    )
    
    total_jobs: int = Field(
        description="Total number of generation jobs"
    )
    
    job_ids: List[str] = Field(
        description="List of individual job IDs"
    )
    
    status_url: str = Field(
        description="URL to check batch status"
    )


# ==========================================
# Evaluation Response Schemas
# ==========================================

class MetricResult(BaseModel):
    """Individual metric result."""
    
    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    threshold: Optional[float] = Field(default=None, description="Threshold if applicable")
    passed: Optional[bool] = Field(default=None, description="Whether threshold was passed")
    description: Optional[str] = Field(default=None, description="Metric description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class FairnessReportResponse(BaseAPIResponse):
    """
    Response model for fairness evaluation.
    
    Contains comprehensive fairness metrics and analysis.
    """
    
    success: Literal[True] = True
    
    job_id: str
    
    # Summary scores
    overall_fairness_score: float = Field(
        description="Overall fairness score (0-1)"
    )
    
    overall_fidelity_score: float = Field(
        description="Overall fidelity score (0-1)"
    )
    
    combined_score: float = Field(
        description="Combined fairness-fidelity score"
    )
    
    # Detailed metrics
    group_fairness_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Group fairness metric results"
    )
    
    individual_fairness_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Individual fairness metric results"
    )
    
    counterfactual_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Counterfactual fairness metric results"
    )
    
    intersectional_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Intersectional fairness metric results"
    )
    
    # Threshold validation
    passed_all_thresholds: bool = Field(
        description="Whether all fairness thresholds were passed"
    )
    
    failed_metrics: List[str] = Field(
        default_factory=list,
        description="Names of metrics that failed threshold validation"
    )
    
    # Privacy metrics (if evaluated)
    privacy_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Privacy evaluation metrics"
    )
    
    privacy_risk_level: Optional[str] = Field(
        default=None,
        description="Overall privacy risk assessment"
    )
    
    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving fairness"
    )
    
    # Report artifact
    report_url: Optional[str] = Field(
        default=None,
        description="URL to detailed HTML report"
    )


class EvaluationResponse(BaseAPIResponse):
    """
    Response model for evaluation requests.
    
    Provides comprehensive evaluation results including
    fairness, fidelity, and privacy assessments.
    """
    
    success: Literal[True] = True
    
    job_id: str
    
    evaluation_time: float = Field(
        description="Time taken for evaluation in seconds"
    )
    
    n_samples_evaluated: int = Field(
        description="Number of samples evaluated"
    )
    
    # Fairness results
    fairness: Optional[FairnessReportResponse] = Field(
        default=None,
        description="Fairness evaluation results"
    )
    
    # Fidelity results
    fidelity_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Fidelity metric results"
    )
    
    statistical_similarity: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistical similarity scores"
    )
    
    distribution_comparison: Dict[str, Any] = Field(
        default_factory=dict,
        description="Distribution comparison results"
    )
    
    # Privacy results
    privacy_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Privacy evaluation metrics"
    )
    
    privacy_risk_assessment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Privacy risk assessment details"
    )
    
    # Multimodal results
    multimodal_metrics: List[MetricResult] = Field(
        default_factory=list,
        description="Cross-modal consistency metrics"
    )
    
    # Report
    report_url: Optional[str] = Field(
        default=None,
        description="URL to detailed evaluation report"
    )


# ==========================================
# Training Response Schemas
# ==========================================

class TrainingProgress(BaseModel):
    """Training progress details."""
    
    current_epoch: int = Field(description="Current epoch")
    total_epochs: int = Field(description="Total epochs")
    current_batch: int = Field(description="Current batch in epoch")
    total_batches: int = Field(description="Total batches per epoch")
    loss: float = Field(description="Current loss value")
    fairness_loss: float = Field(description="Current fairness loss")
    reconstruction_loss: float = Field(description="Current reconstruction loss")
    learning_rate: float = Field(description="Current learning rate")
    eta_seconds: Optional[float] = Field(default=None, description="Estimated time remaining")
    gpu_memory_used: Optional[float] = Field(default=None, description="GPU memory usage in GB")
    samples_per_second: Optional[float] = Field(default=None, description="Training throughput")


class TrainingResponse(BaseAPIResponse):
    """
    Response model for training requests.
    
    Returns job information for async training tasks.
    """
    
    success: Literal[True] = True
    
    job_id: str = Field(description="Training job ID")
    
    model_name: str = Field(description="Name of model being trained")
    
    status: JobStatus = Field(description="Current training status")
    
    message: str = Field(description="Status message")
    
    estimated_time: Optional[float] = Field(
        default=None,
        description="Estimated time to completion in seconds"
    )
    
    status_url: str = Field(description="URL to check training status")
    
    tensorboard_url: Optional[str] = Field(
        default=None,
        description="TensorBoard URL for monitoring"
    )


class TrainingResult(BaseAPIResponse):
    """
    Result of a completed training job.
    
    Contains model artifacts and training metrics.
    """
    
    success: Literal[True] = True
    
    job_id: str
    
    model_name: str
    
    model_path: str = Field(description="Path to trained model")
    
    config_path: str = Field(description="Path to model configuration")
    
    training_time: float = Field(description="Total training time in seconds")
    
    epochs_completed: int = Field(description="Number of epochs completed")
    
    best_epoch: int = Field(description="Epoch with best validation loss")
    
    # Final metrics
    final_loss: float = Field(description="Final training loss")
    
    final_fairness_loss: float = Field(description="Final fairness loss")
    
    best_validation_loss: float = Field(description="Best validation loss")
    
    best_fairness_score: float = Field(description="Best achieved fairness score")
    
    # Model info
    model_size_bytes: int = Field(description="Model file size")
    
    num_parameters: int = Field(description="Total number of parameters")
    
    # Artifacts
    checkpoint_paths: List[str] = Field(
        default_factory=list,
        description="Paths to training checkpoints"
    )
    
    training_log_url: Optional[str] = Field(
        default=None,
        description="URL to training logs"
    )
    
    evaluation_report_url: Optional[str] = Field(
        default=None,
        description="URL to model evaluation report"
    )


# ==========================================
# Health and Status Response Schemas
# ==========================================

class HealthResponse(BaseAPIResponse):
    """
    Health check response.
    
    Provides system health status and component availability.
    """
    
    success: Literal[True] = True
    
    status: HealthStatus = Field(description="Overall health status")
    
    version: str = Field(description="API version")
    
    uptime_seconds: float = Field(description="Server uptime in seconds")
    
    # Component status
    database_status: str = Field(description="Database connection status")
    
    model_registry_status: str = Field(description="Model registry status")
    
    gpu_available: bool = Field(description="Whether GPU is available")
    
    gpu_count: int = Field(default=0, description="Number of available GPUs")
    
    gpu_memory_total: Optional[float] = Field(
        default=None,
        description="Total GPU memory in GB"
    )
    
    gpu_memory_free: Optional[float] = Field(
        default=None,
        description="Free GPU memory in GB"
    )
    
    # System resources
    cpu_usage_percent: float = Field(description="CPU usage percentage")
    
    memory_usage_percent: float = Field(description="Memory usage percentage")
    
    disk_usage_percent: float = Field(description="Disk usage percentage")
    
    # Job statistics
    active_jobs: int = Field(description="Number of active jobs")
    
    queued_jobs: int = Field(description="Number of queued jobs")
    
    completed_jobs_today: int = Field(description="Jobs completed today")


class StatusResponse(BaseAPIResponse):
    """
    Response for job status queries.
    
    Provides detailed job progress and results.
    """
    
    success: Literal[True] = True
    
    job_id: str = Field(description="Job ID")
    
    job_type: str = Field(description="Type of job (generation, evaluation, training)")
    
    status: JobStatus = Field(description="Current job status")
    
    progress_percent: float = Field(
        description="Progress percentage (0-100)"
    )
    
    message: str = Field(description="Current status message")
    
    created_at: datetime = Field(description="Job creation time")
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="Job start time"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion time"
    )
    
    # Progress details
    progress: Optional[TrainingProgress] = Field(
        default=None,
        description="Detailed progress for training jobs"
    )
    
    # Result (if completed)
    result: Optional[Union[GenerationResult, EvaluationResponse, TrainingResult]] = Field(
        default=None,
        description="Job result if completed"
    )
    
    # Error (if failed)
    error: Optional[ErrorResponse] = Field(
        default=None,
        description="Error details if failed"
    )
    
    # Logs
    recent_logs: Optional[List[str]] = Field(
        default=None,
        description="Recent log entries if requested"
    )


class DownloadResponse(BaseAPIResponse):
    """Response for data download requests."""
    
    success: Literal[True] = True
    
    job_id: str
    
    download_url: str = Field(description="Download URL")
    
    file_name: str = Field(description="File name")
    
    file_size_bytes: int = Field(description="File size")
    
    content_type: str = Field(description="MIME type")
    
    checksum: str = Field(description="SHA256 checksum")
    
    expires_at: datetime = Field(description="URL expiration time")


# ==========================================
# Model Management Response Schemas
# ==========================================

class ModelInfoResponse(BaseAPIResponse):
    """
    Response for model information queries.
    
    Provides detailed information about available models.
    """
    
    success: Literal[True] = True
    
    model_name: str = Field(description="Model name")
    
    model_type: str = Field(description="Generator architecture type")
    
    modality: str = Field(description="Data modality")
    
    framework: str = Field(description="Deep learning framework")
    
    description: Optional[str] = Field(default=None, description="Model description")
    
    version: str = Field(description="Model version")
    
    created_at: datetime = Field(description="Model creation date")
    
    trained_epochs: int = Field(description="Number of training epochs")
    
    fairness_type: Optional[str] = Field(
        default=None,
        description="Type of fairness training"
    )
    
    # Performance metrics
    fairness_score: Optional[float] = Field(
        default=None,
        description="Model's fairness score"
    )
    
    fidelity_score: Optional[float] = Field(
        default=None,
        description="Model's fidelity score"
    )
    
    # Model details
    num_parameters: int = Field(description="Total parameters")
    
    model_size_bytes: int = Field(description="Model file size")
    
    tags: List[str] = Field(default_factory=list, description="Model tags")
    
    # Usage
    generation_count: int = Field(description="Number of generations using this model")
    
    last_used: Optional[datetime] = Field(default=None, description="Last usage time")


class ModelListResponse(BaseAPIResponse):
    """Response for listing available models."""
    
    success: Literal[True] = True
    
    models: List[ModelInfoResponse] = Field(description="List of available models")
    
    total_count: int = Field(description="Total number of models")
    
    page: int = Field(description="Current page")
    
    page_size: int = Field(description="Items per page")
