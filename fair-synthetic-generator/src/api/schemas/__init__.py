"""
API Schemas Module
==================

Pydantic models for request/response validation.
"""

from src.api.schemas.request import (
    GenerationRequest,
    EvaluationRequest,
    TrainingRequest,
    ModelUploadRequest,
    BatchGenerationRequest,
    FairnessConfigRequest,
    MultiModalGenerationRequest,
    StatusQueryRequest,
)

from src.api.schemas.response import (
    GenerationResponse,
    EvaluationResponse,
    TrainingResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    BatchGenerationResponse,
    StatusResponse,
    DownloadResponse,
    FairnessReportResponse,
)

__all__ = [
    # Request schemas
    "GenerationRequest",
    "EvaluationRequest",
    "TrainingRequest",
    "ModelUploadRequest",
    "BatchGenerationRequest",
    "FairnessConfigRequest",
    "MultiModalGenerationRequest",
    "StatusQueryRequest",
    # Response schemas
    "GenerationResponse",
    "EvaluationResponse",
    "TrainingResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "ErrorResponse",
    "BatchGenerationResponse",
    "StatusResponse",
    "DownloadResponse",
    "FairnessReportResponse",
]
