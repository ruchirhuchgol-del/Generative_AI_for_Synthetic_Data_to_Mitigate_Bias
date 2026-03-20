"""
Generation Routes
=================

Endpoints for synthetic data generation operations.

Features:
- Single and batch generation endpoints
- Multi-modal generation support
- Job status tracking
- Download and export functionality
- Model selection and configuration
"""

import os
import json
import uuid
import asyncio
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.schemas.request import (
    GenerationRequest,
    BatchGenerationRequest,
    MultiModalGenerationRequest,
    FairnessConfigRequest,
    ModalityType,
    GeneratorType,
    OutputFormat,
)
from src.api.schemas.response import (
    GenerationResponse,
    GenerationResult,
    BatchGenerationResponse,
    StatusResponse,
    DownloadResponse,
    ErrorResponse,
    JobStatus,
)


router = APIRouter(prefix="/generate", tags=["Generation"])


# ==========================================
# In-Memory Job Storage (Replace with Redis/DB in production)
# ==========================================

class JobManager:
    """In-memory job storage and management."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Any] = {}
    
    def create_job(self, job_type: str, config: Dict[str, Any]) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.QUEUED,
            "config": config,
            "progress_percent": 0.0,
            "message": "Job queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, **updates) -> None:
        """Update job fields."""
        if job_id in self._jobs:
            self._jobs[job_id].update(updates)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        return jobs[:limit]


# Global job manager instance
job_manager = JobManager()


# ==========================================
# Background Task Functions
# ==========================================

async def run_generation_job(job_id: str, config: Dict[str, Any]):
    """
    Background task for synthetic data generation.
    
    In production, this would:
    1. Load the specified model
    2. Configure fairness constraints
    3. Generate samples
    4. Run post-processing
    5. Save results
    """
    try:
        # Update job status
        job_manager.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            message="Starting generation..."
        )
        
        n_samples = config.get("n_samples", 1000)
        modality = config.get("modality", "tabular")
        generator_type = config.get("generator_type", "vae")
        
        # Simulate generation progress
        for progress in range(0, 101, 10):
            await asyncio.sleep(0.1)  # Simulate work
            job_manager.update_job(
                job_id,
                progress_percent=float(progress),
                message=f"Generating samples... {progress}%"
            )
        
        # Create mock result
        output_dir = Path("data/synthetic")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{job_id}_synthetic.csv"
        
        # Generate mock data file
        import pandas as pd
        import numpy as np
        
        np.random.seed(config.get("seed", 42))
        
        # Create mock dataframe
        data = {
            "id": range(n_samples),
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.choice(["A", "B", "C"], n_samples),
            "sensitive_attr": np.random.choice([0, 1], n_samples),
            "target": np.random.choice([0, 1], n_samples)
        }
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        # Compute checksum
        with open(output_file, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        file_size = output_file.stat().st_size
        
        # Create result
        result = {
            "job_id": job_id,
            "n_samples_generated": n_samples,
            "generation_time": 5.0,  # Mock time
            "data_url": f"/generate/download/{job_id}",
            "data_format": config.get("output_format", "csv"),
            "data_size_bytes": file_size,
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "quality_score": 0.92,
            "fidelity_score": 0.88,
            "fairness_score": 0.95,
            "report_url": f"/evaluation/report/{job_id}",
            "generator_model": generator_type,
            "seed_used": config.get("seed"),
            "checksum": checksum
        }
        
        # Update job as completed
        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            completed_at=datetime.utcnow(),
            message="Generation completed successfully",
            result=result
        )
        
    except Exception as e:
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow(),
            message=f"Generation failed: {str(e)}",
            error={
                "error_code": "generation_error",
                "error_message": str(e)
            }
        )


# ==========================================
# Routes
# ==========================================

@router.post(
    "",
    response_model=GenerationResponse,
    summary="Generate synthetic data",
    description="Submit a request to generate fair synthetic data."
)
async def generate_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate synthetic data.
    
    Submit a generation request for fair synthetic data. The request is
    processed asynchronously and can be tracked via the status endpoint.
    
    - **n_samples**: Number of synthetic samples to generate
    - **modality**: Data modality (tabular, text, image, multimodal)
    - **generator_type**: Generator architecture to use
    - **fairness_config**: Optional fairness constraint configuration
    """
    # Create job
    config = request.model_dump()
    job_id = job_manager.create_job("generation", config)
    
    # Queue background task
    background_tasks.add_task(run_generation_job, job_id, config)
    
    # Estimate time (rough heuristic)
    n_samples = request.n_samples
    estimated_time = max(5, n_samples / 1000 * 2)
    
    return GenerationResponse(
        request_id=request.request_id,
        job_id=job_id,
        status=JobStatus.QUEUED,
        n_samples=n_samples,
        modality=request.modality.value if isinstance(request.modality, ModalityType) else request.modality,
        message="Generation request accepted. Use /generate/status/{job_id} to track progress.",
        estimated_time=estimated_time,
        status_url=f"/generate/status/{job_id}"
    )


@router.post(
    "/batch",
    response_model=BatchGenerationResponse,
    summary="Batch generation",
    description="Submit multiple generation requests in batch."
)
async def batch_generate(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch generation endpoint.
    
    Submit multiple generation configurations at once. Each configuration
    will be processed as a separate job that can be tracked individually.
    """
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    for config in request.configurations:
        config_dict = config.model_dump()
        job_id = job_manager.create_job("generation", config_dict)
        job_ids.append(job_id)
        background_tasks.add_task(run_generation_job, job_id, config_dict)
    
    return BatchGenerationResponse(
        request_id=request.request_id,
        batch_id=batch_id,
        total_jobs=len(job_ids),
        job_ids=job_ids,
        status_url=f"/generate/batch/{batch_id}/status"
    )


@router.post(
    "/multimodal",
    response_model=GenerationResponse,
    summary="Multi-modal generation",
    description="Generate synthetic data across multiple modalities."
)
async def generate_multimodal(
    request: MultiModalGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Multi-modal generation endpoint.
    
    Generate synthetic data across multiple modalities (e.g., tabular + text,
    or tabular + image) with cross-modal consistency constraints.
    """
    config = request.model_dump()
    config["multimodal"] = True
    job_id = job_manager.create_job("multimodal_generation", config)
    
    background_tasks.add_task(run_generation_job, job_id, config)
    
    return GenerationResponse(
        request_id=request.request_id,
        job_id=job_id,
        status=JobStatus.QUEUED,
        n_samples=request.n_samples,
        modality="multimodal",
        message="Multi-modal generation request accepted.",
        status_url=f"/generate/status/{job_id}"
    )


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Get job status",
    description="Check the status of a generation job."
)
async def get_job_status(job_id: str):
    """
    Get generation job status.
    
    Returns the current status and progress of a generation job.
    If the job is complete, includes download information.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return StatusResponse(
        job_id=job["job_id"],
        job_type=job["job_type"],
        status=job["status"],
        progress_percent=job["progress_percent"],
        message=job["message"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result=job.get("result")
    )


@router.get(
    "/download/{job_id}",
    summary="Download generated data",
    description="Download the generated synthetic data file."
)
async def download_generated_data(job_id: str):
    """
    Download generated data.
    
    Returns the generated synthetic data file. The file is available
    for 24 hours after generation completes.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}"
        )
    
    # Find output file
    output_file = Path(f"data/synthetic/{job_id}_synthetic.csv")
    
    if not output_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Generated data file not found"
        )
    
    return FileResponse(
        path=output_file,
        filename=f"synthetic_data_{job_id}.csv",
        media_type="text/csv"
    )


@router.get(
    "/download/{job_id}/info",
    response_model=DownloadResponse,
    summary="Get download info",
    description="Get download URL and metadata for generated data."
)
async def get_download_info(job_id: str):
    """Get download information including URL, size, and expiration."""
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}"
        )
    
    result = job.get("result", {})
    
    return DownloadResponse(
        job_id=job_id,
        download_url=f"/generate/download/{job_id}",
        file_name=f"synthetic_data_{job_id}.csv",
        file_size_bytes=result.get("data_size_bytes", 0),
        content_type="text/csv",
        checksum=result.get("checksum", ""),
        expires_at=datetime.fromisoformat(result["expires_at"]) if result.get("expires_at") else datetime.utcnow() + timedelta(hours=24)
    )


@router.get(
    "/jobs",
    summary="List generation jobs",
    description="List all generation jobs with optional status filter."
)
async def list_jobs(
    status: Optional[JobStatus] = Query(default=None, description="Filter by job status"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of jobs to return")
):
    """List generation jobs."""
    jobs = job_manager.list_jobs(status=status, limit=limit)
    
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "progress": j["progress_percent"],
                "message": j["message"],
                "created_at": j["created_at"].isoformat(),
                "completed_at": j.get("completed_at").isoformat() if j.get("completed_at") else None
            }
            for j in jobs
        ]
    }


@router.delete(
    "/jobs/{job_id}",
    summary="Cancel job",
    description="Cancel a running generation job."
)
async def cancel_job(job_id: str):
    """Cancel a generation job."""
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    job_manager.update_job(
        job_id,
        status=JobStatus.CANCELLED,
        completed_at=datetime.utcnow(),
        message="Job cancelled by user"
    )
    
    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


@router.get(
    "/models",
    summary="List available models",
    description="List all available pre-trained generator models."
)
async def list_models():
    """
    List available pre-trained models.
    
    Returns a list of models that can be used for generation,
    including their capabilities and performance metrics.
    """
    return {
        "models": [
            {
                "name": "fair-vae-tabular",
                "type": "vae",
                "modalities": ["tabular"],
                "fairness_type": "adversarial",
                "fairness_score": 0.95,
                "fidelity_score": 0.88,
                "parameters": "2.5M",
                "description": "Adversarially debiased VAE for tabular data"
            },
            {
                "name": "fair-wgan-tabular",
                "type": "wgan_gp",
                "modalities": ["tabular"],
                "fairness_type": "constrained",
                "fairness_score": 0.93,
                "fidelity_score": 0.91,
                "parameters": "3.2M",
                "description": "Wasserstein GAN with gradient penalty and fairness constraints"
            },
            {
                "name": "fair-diffusion-multimodal",
                "type": "ldm",
                "modalities": ["tabular", "text", "image"],
                "fairness_type": "constrained",
                "fairness_score": 0.91,
                "fidelity_score": 0.94,
                "parameters": "50M",
                "description": "Latent diffusion model for multi-modal fair generation"
            },
            {
                "name": "counterfactual-vae",
                "type": "vae",
                "modalities": ["tabular"],
                "fairness_type": "counterfactual",
                "fairness_score": 0.97,
                "fidelity_score": 0.85,
                "parameters": "2.8M",
                "description": "VAE with counterfactual fairness constraints"
            }
        ]
    }


@router.post(
    "/preview",
    summary="Preview generation",
    description="Generate a small preview of synthetic data."
)
async def preview_generation(request: GenerationRequest):
    """
    Preview synthetic data generation.
    
    Generates a small sample (up to 100 records) for preview purposes.
    Does not persist the generated data.
    """
    # Limit preview size
    preview_samples = min(request.n_samples, 100)
    
    import numpy as np
    import pandas as pd
    
    np.random.seed(request.seed or 42)
    
    # Generate mock preview data
    data = {
        "id": list(range(preview_samples)),
        "feature_1": np.random.randn(preview_samples).tolist(),
        "feature_2": np.random.randn(preview_samples).tolist(),
        "feature_3": np.random.choice(["A", "B", "C"], preview_samples).tolist(),
        "sensitive_attr": np.random.choice([0, 1], preview_samples).tolist(),
        "target": np.random.choice([0, 1], preview_samples).tolist()
    }
    
    return {
        "preview": True,
        "n_samples": preview_samples,
        "modality": request.modality.value if isinstance(request.modality, ModalityType) else request.modality,
        "generator_type": request.generator_type.value if isinstance(request.generator_type, GeneratorType) else request.generator_type,
        "data": data,
        "message": "This is a preview. Full generation requires POST to /generate endpoint."
    }
