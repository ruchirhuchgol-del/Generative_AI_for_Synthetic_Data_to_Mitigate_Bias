"""
FastAPI Application
===================

Main FastAPI application for fair synthetic data generation.

Features:
- Modular router architecture
- CORS middleware configuration
- Request logging and tracing
- OpenAPI documentation
- Health monitoring endpoints
- Background task processing

Usage:
    # Development
    uvicorn src.api.app:app --reload
    
    # Production
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Programmatic
    from src.api import run_server
    run_server(port=8000)
"""

import os
import time
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

from src.api.routes import health_router, generation_router, evaluation_router
from src.api.middleware.logging_middleware import LoggingMiddleware, LogLevel
from src.api.schemas.response import ErrorResponse


# ==========================================
# Application Configuration
# ==========================================

APP_VERSION = os.environ.get("APP_VERSION", "0.1.0")
APP_NAME = "Fair Synthetic Data Generator API"
APP_DESCRIPTION = """
# Fair Synthetic Data Generator API

A comprehensive REST API for generating fair synthetic data using generative AI.

## Features

- **Multi-Modal Generation**: Support for tabular, text, and image modalities
- **Fairness Constraints**: Group, individual, and counterfactual fairness paradigms
- **Multiple Architectures**: VAE, GAN (WGAN-GP, StyleGAN), and Diffusion models
- **Comprehensive Evaluation**: Fairness, fidelity, and privacy metrics
- **Privacy Protection**: Differential privacy and privacy risk assessment

## Getting Started

1. Use `/generate` to create synthetic data
2. Use `/evaluate` to assess quality and fairness
3. Use `/health` to check system status

## Authentication

API key authentication is required for all endpoints except `/health`.
Include your API key in the `X-API-Key` header.
"""

# Global start time for uptime tracking
START_TIME = time.time()


# ==========================================
# Lifespan Context Manager
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the application.
    """
    # Startup
    print("Starting Fair Synthetic Data Generator API...")
    
    # Initialize resources
    # - Database connections
    # - Model registry
    # - Job queue
    # - Cache
    
    yield
    
    # Shutdown
    print("Shutting down Fair Synthetic Data Generator API...")
    
    # Cleanup resources
    # - Close database connections
    # - Flush logs
    # - Cancel background tasks


# ==========================================
# Application Factory
# ==========================================

def create_app(
    title: str = APP_NAME,
    version: str = APP_VERSION,
    description: str = APP_DESCRIPTION,
    cors_origins: Optional[list] = None,
    enable_logging: bool = True,
    log_level: LogLevel = LogLevel.INFO
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        title: Application title
        version: Application version
        description: Application description
        cors_origins: List of allowed CORS origins
        enable_logging: Whether to enable request logging
        log_level: Logging level
        
    Returns:
        Configured FastAPI application instance
    """
    # Create app
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            422: {"model": ErrorResponse, "description": "Validation Error"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            503: {"model": ErrorResponse, "description": "Service Unavailable"},
        }
    )
    
    # CORS middleware
    origins = cors_origins or os.environ.get(
        "CORS_ORIGINS", 
        "*"
    ).split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    if enable_logging:
        app.add_middleware(
            LoggingMiddleware,
            service_name="fair-synthetic-api",
            log_level=log_level,
            log_request_body=False,
            log_response_body=False,
            exclude_paths=["/health", "/metrics", "/favicon.ico"],
            json_format=True
        )
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(generation_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect to docs."""
        return {
            "name": APP_NAME,
            "version": version,
            "docs": "/docs",
            "health": "/api/v1/health",
            "openapi": "/openapi.json"
        }
    
    # API info endpoint
    @app.get("/api/v1", tags=["API Info"])
    async def api_info():
        """Get API information and available endpoints."""
        return {
            "name": APP_NAME,
            "version": version,
            "endpoints": {
                "health": "/api/v1/health",
                "generate": "/api/v1/generate",
                "evaluate": "/api/v1/evaluate",
                "models": "/api/v1/generate/models",
                "metrics": "/api/v1/evaluate/metrics"
            }
        }
    
    # Global exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error response format."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_code": exc.detail.get("code", "error") if isinstance(exc.detail, dict) else "error",
                "error_message": exc.detail.get("message", str(exc.detail)) if isinstance(exc.detail, dict) else str(exc.detail),
                "timestamp": time.time()
            }
        )
    
    # Generic exception handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "internal_error",
                "error_message": "An unexpected error occurred",
                "details": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc()
                } if os.environ.get("DEBUG") == "true" else None,
                "timestamp": time.time()
            }
        )
    
    return app


# ==========================================
# Create Default App Instance
# ==========================================

app = create_app()


# ==========================================
# Custom OpenAPI Schema
# ==========================================

def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        routes=app.routes,
        tags=[
            {
                "name": "Health",
                "description": "Health check and monitoring endpoints."
            },
            {
                "name": "Generation",
                "description": "Synthetic data generation operations."
            },
            {
                "name": "Evaluation",
                "description": "Fairness and fidelity evaluation operations."
            },
            {
                "name": "API Info",
                "description": "API information and metadata."
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ==========================================
# Server Runner
# ==========================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    print(f"""
    +--------------------------------------------------------------+
    |         Fair Synthetic Data Generator API                    |
    +--------------------------------------------------------------+
    |  Version: {APP_VERSION:<49} |
    |  Host: {host:<52} |
    |  Port: {port:<52} |
    |  Workers: {workers:<50} |
    |  Docs: http://{host}:{port}/docs{' ' * (42 - len(host) - len(str(port)))} |
    +--------------------------------------------------------------+
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


# ==========================================
# Main Entry Point
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fair Synthetic Data Generator API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )
