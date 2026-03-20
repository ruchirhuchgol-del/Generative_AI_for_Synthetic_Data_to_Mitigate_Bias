"""
API Module
==========

REST API for synthetic data generation service.

This module provides a comprehensive FastAPI-based REST API for:
- Fair synthetic data generation (single/batch/multi-modal)
- Fairness and fidelity evaluation
- Model management and discovery
- Job status tracking
- Health monitoring

Architecture:
- app.py: Main FastAPI application and configuration
- routes/: API endpoint handlers
- schemas/: Request/response Pydantic models
- middleware/: Request processing middleware

Usage:
    from src.api import app, run_server
    
    # Run the server
    run_server(host="0.0.0.0", port=8000)
    
    # Or use with uvicorn directly
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from src.api.app import app, run_server, create_app

__all__ = [
    "app",
    "run_server",
    "create_app",
]
