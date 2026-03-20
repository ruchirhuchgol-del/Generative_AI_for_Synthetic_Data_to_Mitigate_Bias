"""
Logging Middleware
==================

Comprehensive request/response logging middleware for FastAPI.

Features:
- Request/response logging with timing
- Structured JSON logging
- Request ID tracking
- Error logging with stack traces
- Sensitive data redaction
- Performance metrics collection
- Integration with distributed tracing

Usage:
    from src.api.middleware.logging_middleware import LoggingMiddleware
    
    app = FastAPI()
    app.add_middleware(LoggingMiddleware, service_name="fair-synthetic-api")
"""

import time
import uuid
import json
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
import asyncio


# ==========================================
# Context Variables for Request Tracking
# ==========================================

request_id_context: ContextVar[str] = ContextVar("request_id", default="")
request_start_time: ContextVar[float] = ContextVar("request_start_time", default=0.0)


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RequestLog:
    """Structured request log entry."""
    
    request_id: str
    method: str
    path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    client_ip: str
    user_agent: str
    request_body: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Response fields
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    response_size: Optional[int] = None
    
    # Error fields
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None


class RequestLogger:
    """
    Structured logger for API requests.
    
    Provides JSON-formatted logging with request context,
    sensitive data redaction, and performance tracking.
    """
    
    # Fields to redact for security
    SENSITIVE_FIELDS: Set[str] = {
        "password", "passwd", "pwd", "secret", "token", "api_key",
        "apikey", "authorization", "auth", "credential", "private_key",
        "access_token", "refresh_token", "session_id", "cookie"
    }
    
    # Headers to redact
    SENSITIVE_HEADERS: Set[str] = {
        "authorization", "cookie", "set-cookie", "x-api-key",
        "x-auth-token", "x-access-token"
    }
    
    def __init__(
        self,
        service_name: str = "fair-synthetic-api",
        log_level: LogLevel = LogLevel.INFO,
        redact_sensitive: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 10000,
        exclude_paths: Optional[List[str]] = None,
        json_format: bool = True
    ):
        """
        Initialize the request logger.
        
        Args:
            service_name: Name of the service for log identification
            log_level: Minimum log level to record
            redact_sensitive: Whether to redact sensitive fields
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            max_body_size: Maximum body size to log (chars)
            exclude_paths: Paths to exclude from logging
            json_format: Whether to use JSON format for logs
        """
        self.service_name = service_name
        self.log_level = log_level
        self.redact_sensitive = redact_sensitive
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.json_format = json_format
        
        # Configure logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Add handler if not present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, log_level.value))
            
            if json_format:
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive fields from a dictionary.
        
        Args:
            data: Dictionary to redact
            
        Returns:
            Dictionary with sensitive values replaced by [REDACTED]
        """
        if not self.redact_sensitive:
            return data
        
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            if key_lower in self.SENSITIVE_FIELDS:
                result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive headers."""
        if not self.redact_sensitive:
            return headers
        
        return {
            key: "[REDACTED]" if key.lower() in self.SENSITIVE_HEADERS else value
            for key, value in headers.items()
        }
    
    def truncate_body(self, body: str) -> str:
        """Truncate body if too large."""
        if len(body) > self.max_body_size:
            return body[:self.max_body_size] + "... [TRUNCATED]"
        return body
    
    def format_log(self, log_data: Dict[str, Any]) -> str:
        """Format log entry."""
        if self.json_format:
            return json.dumps(log_data, default=str)
        else:
            # Human-readable format
            parts = []
            for key, value in log_data.items():
                parts.append(f"{key}={value}")
            return " | ".join(parts)
    
    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, str],
        client_ip: str,
        request_body: Optional[str] = None
    ) -> None:
        """Log incoming request."""
        log_data = {
            "event": "request",
            "service": self.service_name,
            "request_id": request_id,
            "method": method,
            "path": path,
            "query_params": self.redact_dict(query_params) if query_params else {},
            "headers": self.redact_headers(headers),
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.log_request_body and request_body:
            try:
                body_json = json.loads(request_body)
                log_data["request_body"] = self.truncate_body(
                    json.dumps(self.redact_dict(body_json))
                )
            except json.JSONDecodeError:
                log_data["request_body"] = self.truncate_body(request_body)
        
        self.logger.info(self.format_log(log_data))
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time_ms: float,
        response_size: int,
        response_body: Optional[str] = None
    ) -> None:
        """Log outgoing response."""
        log_data = {
            "event": "response",
            "service": self.service_name,
            "request_id": request_id,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
            "response_size_bytes": response_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.log_response_body and response_body:
            try:
                body_json = json.loads(response_body)
                log_data["response_body"] = self.truncate_body(
                    json.dumps(self.redact_dict(body_json))
                )
            except json.JSONDecodeError:
                log_data["response_body"] = self.truncate_body(response_body)
        
        # Use appropriate log level based on status code
        if status_code >= 500:
            self.logger.error(self.format_log(log_data))
        elif status_code >= 400:
            self.logger.warning(self.format_log(log_data))
        else:
            self.logger.info(self.format_log(log_data))
    
    def log_error(
        self,
        request_id: str,
        error: Exception,
        path: str,
        method: str
    ) -> None:
        """Log error with stack trace."""
        log_data = {
            "event": "error",
            "service": self.service_name,
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_stack": traceback.format_exc(),
            "path": path,
            "method": method,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.error(self.format_log(log_data))


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for comprehensive request/response logging.
    
    Features:
    - Automatic request ID generation
    - Request timing
    - Structured JSON logging
    - Error capture and logging
    - Configurable sensitive data redaction
    
    Example:
        app = FastAPI()
        app.add_middleware(
            LoggingMiddleware,
            service_name="my-api",
            log_request_body=True,
            exclude_paths=["/health"]
        )
    """
    
    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "fair-synthetic-api",
        log_level: LogLevel = LogLevel.INFO,
        log_request_body: bool = False,
        log_response_body: bool = False,
        redact_sensitive: bool = True,
        exclude_paths: Optional[List[str]] = None,
        json_format: bool = True
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: ASGI application
            service_name: Service identifier for logs
            log_level: Minimum log level
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            redact_sensitive: Whether to redact sensitive data
            exclude_paths: Paths to exclude from logging
            json_format: Whether to use JSON format
        """
        super().__init__(app)
        
        self.logger = RequestLogger(
            service_name=service_name,
            log_level=log_level,
            log_request_body=log_request_body,
            log_response_body=log_response_body,
            redact_sensitive=redact_sensitive,
            exclude_paths=exclude_paths,
            json_format=json_format
        )
    
    def _should_log(self, path: str) -> bool:
        """Check if path should be logged."""
        for excluded in self.logger.exclude_paths:
            if path.startswith(excluded):
                return False
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request with logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler
            
        Returns:
            HTTP response
        """
        # Generate or extract request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request_id_context.set(request_id)
        
        # Record start time
        start_time = time.time()
        request_start_time.set(start_time)
        
        path = request.url.path
        method = request.method
        
        # Skip logging for excluded paths
        if not self._should_log(path):
            response = await call_next(request)
            return response
        
        # Get request details
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        client_ip = self._get_client_ip(request)
        user_agent = headers.get("user-agent", "unknown")
        
        # Get request body if needed
        request_body = None
        if self.logger.log_request_body and method in ("POST", "PUT", "PATCH"):
            try:
                request_body = (await request.body()).decode("utf-8")
                # Reconstruct request body for downstream handlers
                async def receive():
                    return {"type": "http.request", "body": request_body.encode()}
                request._receive = receive
            except Exception:
                request_body = None
        
        # Log request
        self.logger.log_request(
            request_id=request_id,
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            client_ip=client_ip,
            request_body=request_body
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate timing
            response_time_ms = (time.time() - start_time) * 1000
            
            # Get response body if needed
            response_body = None
            if self.logger.log_response_body:
                try:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    # Create new response with captured body
                    response = Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                    response_body = response_body.decode("utf-8")
                except Exception:
                    pass
            
            # Log response
            self.logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                response_size=int(response.headers.get("content-length", 0)),
                response_body=response_body
            )
            
            # Add request ID to response headers
            response.headers["x-request-id"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate timing for error
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log error
            self.logger.log_error(
                request_id=request_id,
                error=e,
                path=path,
                method=method
            )
            
            # Return error response
            error_response = {
                "success": False,
                "request_id": request_id,
                "error_code": "internal_error",
                "error_message": "An internal server error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = JSONResponse(
                status_code=500,
                content=error_response
            )
            response.headers["x-request-id"] = request_id
            
            return response


def setup_request_logging(
    app,
    service_name: str = "fair-synthetic-api",
    log_level: LogLevel = LogLevel.INFO,
    log_request_body: bool = False,
    log_response_body: bool = False,
    exclude_paths: Optional[List[str]] = None
) -> None:
    """
    Setup request logging middleware with default configuration.
    
    Args:
        app: FastAPI application
        service_name: Service identifier
        log_level: Logging level
        log_request_body: Whether to log request bodies
        log_response_body: Whether to log response bodies
        exclude_paths: Paths to exclude from logging
    """
    app.add_middleware(
        LoggingMiddleware,
        service_name=service_name,
        log_level=log_level,
        log_request_body=log_request_body,
        log_response_body=log_response_body,
        exclude_paths=exclude_paths or ["/health", "/metrics"]
    )


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_context.get()


def get_request_duration() -> float:
    """Get current request duration in seconds."""
    start = request_start_time.get()
    if start > 0:
        return time.time() - start
    return 0.0
