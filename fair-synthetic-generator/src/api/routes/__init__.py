"""
API Routes Module
=================

Route handlers for different API endpoints.
"""

from src.api.routes.health import router as health_router
from src.api.routes.generation import router as generation_router
from src.api.routes.evaluation import router as evaluation_router

__all__ = [
    "health_router",
    "generation_router",
    "evaluation_router",
]
