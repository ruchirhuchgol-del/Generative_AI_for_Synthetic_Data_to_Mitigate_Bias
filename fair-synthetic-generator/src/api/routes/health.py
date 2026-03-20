"""
Health Check Routes
===================

Endpoints for system health monitoring and status checks.

Features:
- Basic health check for load balancers
- Detailed health status with component checks
- System metrics and resource usage
- Readiness and liveness probes for Kubernetes
"""

import os
import time
import platform
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.schemas.response import (
    HealthResponse,
    HealthStatus,
    BaseAPIResponse,
)


router = APIRouter(prefix="/health", tags=["Health"])


# ==========================================
# Response Models
# ==========================================

class ComponentHealth(BaseModel):
    """Health status of a system component."""
    
    name: str = Field(description="Component name")
    status: HealthStatus = Field(description="Component health status")
    message: Optional[str] = Field(default=None, description="Status message")
    latency_ms: Optional[float] = Field(default=None, description="Check latency in ms")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class DetailedHealthResponse(BaseAPIResponse):
    """Detailed health check response with all component statuses."""
    
    success: bool = True
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
    system: Dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseAPIResponse):
    """Response for Kubernetes readiness probe."""
    
    success: bool
    status: str
    checks: Dict[str, bool]


class LivenessResponse(BaseAPIResponse):
    """Response for Kubernetes liveness probe."""
    
    success: bool
    status: str


# ==========================================
# Global State
# ==========================================

START_TIME = time.time()
VERSION = "0.1.0"


# ==========================================
# Health Check Functions
# ==========================================

def check_database() -> ComponentHealth:
    """Check database connectivity."""
    try:
        import sqlite3
        
        start = time.time()
        
        # Try to connect to database
        db_path = os.environ.get("DATABASE_PATH", "data/fair_synthetic.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1")
            conn.close()
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                latency_ms=latency
            )
        else:
            return ComponentHealth(
                name="database",
                status=HealthStatus.DEGRADED,
                message="Database file not found, using in-memory mode"
            )
            
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database check failed: {str(e)}"
        )


def check_model_registry() -> ComponentHealth:
    """Check model registry status."""
    try:
        artifacts_path = os.environ.get("ARTIFACTS_PATH", "artifacts/models")
        
        if os.path.exists(artifacts_path):
            models = [f for f in os.listdir(artifacts_path) if f.endswith(('.pt', '.pth', '.h5'))]
            return ComponentHealth(
                name="model_registry",
                status=HealthStatus.HEALTHY,
                message=f"Model registry accessible, {len(models)} models available",
                details={"model_count": len(models)}
            )
        else:
            return ComponentHealth(
                name="model_registry",
                status=HealthStatus.DEGRADED,
                message="Model registry path not found"
            )
            
    except Exception as e:
        return ComponentHealth(
            name="model_registry",
            status=HealthStatus.UNHEALTHY,
            message=f"Model registry check failed: {str(e)}"
        )


def check_gpu() -> ComponentHealth:
    """Check GPU availability and status."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total_memory - reserved
            
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.HEALTHY,
                message=f"GPU available: {device_name}",
                details={
                    "device_count": device_count,
                    "device_name": device_name,
                    "total_memory_gb": round(total_memory, 2),
                    "free_memory_gb": round(free, 2),
                    "allocated_memory_gb": round(allocated, 2)
                }
            )
        else:
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.DEGRADED,
                message="No GPU available, using CPU"
            )
            
    except ImportError:
        return ComponentHealth(
            name="gpu",
            status=HealthStatus.DEGRADED,
            message="PyTorch not available"
        )
    except Exception as e:
        return ComponentHealth(
            name="gpu",
            status=HealthStatus.UNHEALTHY,
            message=f"GPU check failed: {str(e)}"
        )


def get_system_metrics() -> Dict[str, Any]:
    """Get system resource usage metrics."""
    metrics = {}
    
    try:
        import psutil
        
        # CPU usage
        metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)
        metrics["cpu_count"] = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_usage_percent"] = memory.percent
        metrics["memory_total_gb"] = round(memory.total / (1024**3), 2)
        metrics["memory_available_gb"] = round(memory.available / (1024**3), 2)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics["disk_usage_percent"] = disk.percent
        metrics["disk_total_gb"] = round(disk.total / (1024**3), 2)
        metrics["disk_free_gb"] = round(disk.free / (1024**3), 2)
        
    except ImportError:
        metrics["error"] = "psutil not available"
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics


def get_job_stats() -> Dict[str, int]:
    """Get job queue statistics."""
    # In a real implementation, this would query the job queue
    return {
        "active_jobs": 0,
        "queued_jobs": 0,
        "completed_jobs_today": 0
    }


# ==========================================
# Routes
# ==========================================

@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status. Suitable for load balancer health checks."
)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns a simple health status indicating if the service is running.
    This endpoint is designed for load balancer health checks and should
    be lightweight and fast.
    """
    uptime = time.time() - START_TIME
    
    # Quick checks
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        version=VERSION,
        uptime_seconds=uptime,
        database_status="connected",
        model_registry_status="available",
        gpu_available=gpu_available,
        gpu_count=1 if gpu_available else 0,
        cpu_usage_percent=0.0,
        memory_usage_percent=0.0,
        disk_usage_percent=0.0,
        active_jobs=0,
        queued_jobs=0,
        completed_jobs_today=0
    )


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Returns comprehensive health status including all components."
)
async def detailed_health_check():
    """
    Detailed health check with all component statuses.
    
    Performs thorough health checks of all system components including:
    - Database connectivity
    - Model registry
    - GPU availability
    - System resources
    """
    uptime = time.time() - START_TIME
    
    # Check all components
    components = {
        "database": check_database(),
        "model_registry": check_model_registry(),
        "gpu": check_gpu()
    }
    
    # Determine overall status
    if any(c.status == HealthStatus.UNHEALTHY for c in components.values()):
        overall_status = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in components.values()):
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY
    
    # Get system metrics
    system = get_system_metrics()
    system["platform"] = platform.system()
    system["python_version"] = platform.python_version()
    
    return DetailedHealthResponse(
        success=overall_status != HealthStatus.UNHEALTHY,
        status=overall_status,
        version=VERSION,
        uptime_seconds=uptime,
        components=components,
        system=system
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint."
)
async def readiness_probe():
    """
    Readiness probe for Kubernetes.
    
    Returns whether the service is ready to accept traffic.
    Checks critical dependencies.
    """
    checks = {}
    
    # Check critical components
    db_health = check_database()
    checks["database"] = db_health.status != HealthStatus.UNHEALTHY
    
    # Service is ready if critical checks pass
    ready = all(checks.values())
    
    return ReadinessResponse(
        success=ready,
        status="ready" if ready else "not_ready",
        checks=checks
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint."
)
async def liveness_probe():
    """
    Liveness probe for Kubernetes.
    
    Returns whether the service is alive and running.
    If this returns 200, the container should not be restarted.
    """
    return LivenessResponse(
        success=True,
        status="alive"
    )


@router.get(
    "/metrics",
    summary="System metrics",
    description="Returns detailed system and application metrics."
)
async def get_metrics():
    """
    Get detailed system metrics.
    
    Returns comprehensive metrics including:
    - System resource usage
    - GPU metrics
    - Application metrics
    - Job queue statistics
    """
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - START_TIME,
        "version": VERSION,
        "system": get_system_metrics(),
        "jobs": get_job_stats()
    }
    
    # Add GPU metrics if available
    gpu_health = check_gpu()
    if gpu_health.details:
        metrics["gpu"] = gpu_health.details
    
    return metrics


@router.get(
    "/version",
    summary="Version information",
    description="Returns service version and build information."
)
async def get_version():
    """Get service version and build information."""
    return {
        "version": VERSION,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "build_time": os.environ.get("BUILD_TIME", "unknown"),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "git_branch": os.environ.get("GIT_BRANCH", "unknown")
    }
