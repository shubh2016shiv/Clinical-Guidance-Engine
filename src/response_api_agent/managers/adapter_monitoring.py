"""
Monitoring and logging utilities for the LLM Provider Adapter.

This module provides functions and classes to monitor and log the usage of the
ResponseAPIAdapter, tracking metrics like success rates, response times, and fallbacks.
"""

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps
import asyncio
import datetime

from src.logs import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterMetrics:
    """Metrics for adapter usage."""

    # Request counts
    total_requests: int = 0
    successful_provider_requests: int = 0
    fallback_requests: int = 0
    failed_requests: int = 0

    # Streaming metrics
    streaming_requests: int = 0
    streaming_success: int = 0
    streaming_fallback: int = 0

    # Response times (milliseconds)
    provider_response_times: list = field(default_factory=list)
    fallback_response_times: list = field(default_factory=list)

    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)

    # Model usage
    model_usage: Dict[str, int] = field(default_factory=dict)

    def add_request(
        self,
        provider_used: bool,
        streaming: bool,
        response_time_ms: float,
        error: Optional[Exception] = None,
        model: Optional[str] = None,
    ):
        """Add a request to the metrics."""
        self.total_requests += 1

        # Track streaming
        if streaming:
            self.streaming_requests += 1

        # Track provider vs fallback
        if provider_used and error is None:
            self.successful_provider_requests += 1
            self.provider_response_times.append(response_time_ms)
            if streaming:
                self.streaming_success += 1
        elif error is None:
            self.fallback_requests += 1
            self.fallback_response_times.append(response_time_ms)
            if streaming:
                self.streaming_fallback += 1
        else:
            self.failed_requests += 1
            error_type = type(error).__name__
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        # Track model usage
        if model:
            self.model_usage[model] = self.model_usage.get(model, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics."""
        provider_avg_time = 0
        fallback_avg_time = 0

        if self.provider_response_times:
            provider_avg_time = sum(self.provider_response_times) / len(
                self.provider_response_times
            )

        if self.fallback_response_times:
            fallback_avg_time = sum(self.fallback_response_times) / len(
                self.fallback_response_times
            )

        provider_success_rate = 0
        if self.total_requests > 0:
            provider_success_rate = (
                self.successful_provider_requests / self.total_requests
            ) * 100

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_requests": self.total_requests,
            "provider_success_rate": f"{provider_success_rate:.2f}%",
            "fallback_rate": f"{(self.fallback_requests / self.total_requests * 100) if self.total_requests > 0 else 0:.2f}%",
            "failed_rate": f"{(self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0:.2f}%",
            "streaming_requests": self.streaming_requests,
            "avg_provider_response_time_ms": f"{provider_avg_time:.2f}",
            "avg_fallback_response_time_ms": f"{fallback_avg_time:.2f}",
            "error_types": self.error_types,
            "model_usage": self.model_usage,
        }

    def log_summary(self):
        """Log a summary of the metrics."""
        summary = self.get_summary()
        logger.info(
            "Adapter Metrics Summary",
            component="AdapterMonitoring",
            subcomponent="MetricsSummary",
            **summary,
        )

    def reset(self):
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_provider_requests = 0
        self.fallback_requests = 0
        self.failed_requests = 0
        self.streaming_requests = 0
        self.streaming_success = 0
        self.streaming_fallback = 0
        self.provider_response_times = []
        self.fallback_response_times = []
        self.error_types = {}
        self.model_usage = {}


# Global metrics instance
_metrics = AdapterMetrics()


def get_metrics() -> AdapterMetrics:
    """Get the global metrics instance."""
    global _metrics
    return _metrics


def log_adapter_metrics(
    provider_used: bool,
    streaming: bool,
    response_time_ms: float,
    error: Optional[Exception] = None,
    model: Optional[str] = None,
):
    """Log metrics for adapter usage."""
    get_metrics().add_request(provider_used, streaming, response_time_ms, error, model)


def monitor_adapter_call(func: Callable) -> Callable:
    """Decorator to monitor adapter calls."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _metrics_reporting_started, _metrics_reporting_task

        # Start metrics reporting if it hasn't been started yet
        # (this can happen if start_metrics_reporting() was called without a running event loop)
        if not _metrics_reporting_started:
            try:
                loop = asyncio.get_running_loop()
                # Create task and store reference to prevent duplicate creation
                _metrics_reporting_task = loop.create_task(schedule_metrics_reporting())
                _metrics_reporting_started = True
                logger.debug("Started deferred metrics reporting in decorator")
            except RuntimeError:
                # Still no loop - should not happen in async context, but handle gracefully
                logger.warning(
                    "Cannot start metrics reporting - no event loop available"
                )

        start_time = time.time()
        model = kwargs.get("model")
        streaming = kwargs.get("stream", False)

        try:
            # Call the function (it will initialize provider itself if needed)
            result = await func(*args, **kwargs)

            # Determine provider_used after function execution
            # by checking if provider is available
            self = args[0]
            provider_used = self.llm_provider is not None

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log metrics
            log_adapter_metrics(provider_used, streaming, response_time_ms, model=model)

            return result

        except Exception as e:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Check provider status on error too
            self = args[0]
            provider_used = self.llm_provider is not None

            # Log metrics with error
            log_adapter_metrics(
                provider_used, streaming, response_time_ms, error=e, model=model
            )

            # Re-raise the exception
            raise

    return wrapper


async def schedule_metrics_reporting(interval_seconds: int = 3600):
    """Schedule periodic metrics reporting."""
    while True:
        await asyncio.sleep(interval_seconds)
        get_metrics().log_summary()

        # Optionally reset metrics after reporting
        # get_metrics().reset()


# Global flag and task reference to track if metrics reporting has been started
_metrics_reporting_started = False
_metrics_reporting_task = None


def start_metrics_reporting():
    """Start metrics reporting in the background.

    This function can be called from synchronous contexts (like __init__).
    If an event loop is running, it will create the task immediately.
    If no event loop is running, it will defer task creation to the first
    async operation (via the decorator).
    """
    global _metrics_reporting_started, _metrics_reporting_task

    # If already started, don't create another task
    if _metrics_reporting_started:
        return

    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        # Event loop is running, create task immediately
        _metrics_reporting_task = loop.create_task(schedule_metrics_reporting())
        _metrics_reporting_started = True
        logger.debug("Started metrics reporting in existing event loop")
    except RuntimeError:
        # No event loop running - defer to first async call
        # This is safe because we're in a synchronous context (like __init__)
        # The task will be created when the first decorated async method runs
        _metrics_reporting_started = False
        logger.debug("Deferred metrics reporting start - no event loop running")
