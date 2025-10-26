"""
Enhanced structured logging module for the Drug Recommendation Chatbot.

This module provides structured logging with component naming and execution timing.
It supports both JSON output for file logs and colored output for console logs.
"""

import logging
import structlog
import warnings
import os
import time
import functools
import contextlib
import asyncio
from typing import Optional, Callable, Any, Dict
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Suppress specific warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dotenv")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="dotenv")

# Ensure logs directory exists
logs_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
os.makedirs(logs_dir, exist_ok=True)

# Create log file name with timestamp
log_file = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Global flag to prevent multiple initializations
_logging_configured = False

# ANSI color codes for console output
COLORS = {
    'blue': '\033[34;1m',      # Blue bold for stage numbers
    'orange': '\033[33m',      # Orange/Amber for stage names
    'green': '\033[32m',       # Green for decision fields and success
    'red': '\033[31m',         # Red for failures and errors
    'purple': '\033[35m',      # Purple for config
    'magenta': '\033[35m',     # Magenta for durations/latency
    'gray': '\033[2;37m',      # Dim gray for correlation IDs
    'yellow': '\033[33m',      # Yellow for warnings
    'reset': '\033[0m',        # Reset color
}

# Icons for accessibility
ICONS = {
    'success': '✓',
    'fail': '✗',
    'warning': '!'
}

def add_component_context(_, __, event_dict):
    """
    Add component context to log events.
    
    Creates a formatted component path like [Component > SubComponent]
    based on component and subcomponent fields.
    """
    if 'component' in event_dict and 'subcomponent' in event_dict:
        event_dict['component_path'] = f"[{event_dict['component']} > {event_dict['subcomponent']}]"
    elif 'component' in event_dict:
        event_dict['component_path'] = f"[{event_dict['component']}]"
    return event_dict

def colorize_console_output(_, __, event_dict):
    """
    Format log events with colors for console output.
    
    Formats structured log events into a human-readable colored string
    following the user's color preferences.
    
    This function works with ProcessorFormatter and returns a formatted string.
    """
    # Create a copy to avoid modifying the original
    event_data = event_dict.copy()
    
    # Extract standard fields
    timestamp = event_data.get('timestamp', '')
    level = event_data.get('level', '').upper()
    event = event_data.get('event', '')
    
    # Start building the output
    output_parts = []
    
    # Add timestamp and level
    output_parts.append(f"{timestamp} [{level}]")
    
    # Add component path with color if available
    if 'component_path' in event_data:
        component_path = event_data['component_path']
        output_parts.append(f"{COLORS['blue']}{component_path}{COLORS['reset']}")
    
    # Add the main event message
    output_parts.append(event)
    
    # Add execution time with color if available
    if 'execution_time' in event_data:
        execution_time = event_data['execution_time']
        output_parts.append(f"{COLORS['magenta']}(took {execution_time}){COLORS['reset']}")
    
    # Process remaining fields with appropriate colors
    # Skip standard fields we've already processed
    skip_keys = {'timestamp', 'level', 'event', 'component_path', 'execution_time', 'exc_info', 'exception'}
    
    for key, value in event_data.items():
        if key in skip_keys:
            continue
        
        # Apply specific colors based on field type
        if key == 'status' and value == 'success':
            output_parts.append(f"{key}={COLORS['green']}{ICONS['success']} {value}{COLORS['reset']}")
        elif key == 'status' and (value == 'failed' or value == 'error'):
            output_parts.append(f"{key}={COLORS['red']}{ICONS['fail']} {value}{COLORS['reset']}")
        elif key == 'status' and value == 'warning':
            output_parts.append(f"{key}={COLORS['yellow']}{ICONS['warning']} {value}{COLORS['reset']}")
        elif key == 'error' or key == 'exception_message':
            output_parts.append(f"{key}={COLORS['red']}{value}{COLORS['reset']}")
        elif key == 'warning':
            output_parts.append(f"{key}={COLORS['yellow']}{value}{COLORS['reset']}")
        elif key.endswith('_id'):
            output_parts.append(f"{key}={COLORS['gray']}{value}{COLORS['reset']}")
        else:
            output_parts.append(f"{key}={value}")
    
    # Join all parts with spaces
    return " ".join(output_parts)

def _configure_logging_once():
    """
    Configure logging only once to prevent duplicate handlers.
    This function ensures that logging is configured only once
    regardless of how many times it's called.
    
    Uses ProcessorFormatter for dual output:
    - Console: Colored, human-readable format
    - File: Pure JSON format for structured logging
    """
    global _logging_configured

    if _logging_configured:
        return

    # Configure structlog with shared processors (no final renderer in global config)
    # The final rendering is done by the ProcessorFormatter attached to each handler
    structlog.configure(
        processors=[
            # This allow logger to have a threadlocal context
            structlog.threadlocal.merge_threadlocal_context,
            # This performs the initial filtering, so we don't evaluate e.g. DEBUG when unnecessary
            structlog.stdlib.filter_by_level,
            # Adds level=info, debug, etc.
            structlog.stdlib.add_log_level,
            # Add timestamps
            structlog.processors.TimeStamper(fmt="iso"),
            # Add component context (creates component_path field)
            add_component_context,
            # Performs the % string interpolation as expected
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Include the stack when stack_info=True
            structlog.processors.StackInfoRenderer(),
            # Include the exception when exc_info=True
            structlog.processors.format_exc_info,
            # Decodes the unicode values in any kv pairs
            structlog.processors.UnicodeDecoder(),
            # Prepare for standard logging to process (wrap for ProcessorFormatter)
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=colorize_console_output,
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_component_context,
        ],
    )
    console_handler.setFormatter(console_formatter)

    # Create file handler with JSON formatter
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_component_context,
        ],
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence specific noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.reloader").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.watcher").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("azure.response_api_agent.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.keyvault").setLevel(logging.WARNING)
    logging.getLogger("azure.appconfiguration").setLevel(logging.WARNING)

    # Mark as configured
    _logging_configured = True


def get_logger(log_name: str = __name__) -> structlog._config.BoundLoggerLazyProxy:
    """
    Get a configured logger that logs to both console and file.

    Args:
        log_name: Logger name

    Returns:
        Configured structlog logger
    """
    # Ensure logging is configured only once
    _configure_logging_once()

    # Create and return logger
    logger = structlog.wrap_logger(logging.getLogger(log_name))
    logger.setLevel(logging.INFO)
    return logger


def get_component_logger(component: str) -> structlog._config.BoundLoggerLazyProxy:
    """
    Get a logger pre-configured with a component name.
    
    Args:
        component: Component name to bind to the logger
        
    Returns:
        Logger with component name bound
    """
    logger = get_logger(component)
    return logger.bind(component=component)


@contextlib.contextmanager
def log_execution_time(logger, component: str, operation: str):
    """
    Context manager to log execution time of a block of code.
    
    Args:
        logger: Logger instance to use
        component: Component name
        operation: Operation name (subcomponent)
        
    Example:
        with log_execution_time(logger, "VectorStore", "GetVectorStore"):
            # Code to time
    """
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.info(
            f"Operation completed",
            component=component,
            subcomponent=operation,
            execution_time=f"{execution_time:.3f}s"
        )


def time_execution(component: str, operation: str):
    """
    Decorator to log execution time of a function.
    
    Args:
        component: Component name
        operation: Operation name (subcomponent)
        
    Example:
        @time_execution("VectorStore", "GetVectorStore")
        async def get_vector_store(self, vector_store_id: str):
            # Function code
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get logger from self if available, otherwise create one
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = get_component_logger(component)
                
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                execution_time = time.time() - start_time
                logger.info(
                    f"{func.__name__} completed",
                    component=component,
                    subcomponent=operation,
                    execution_time=f"{execution_time:.3f}s"
                )
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get logger from self if available, otherwise create one
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = get_component_logger(component)
                
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                execution_time = time.time() - start_time
                logger.info(
                    f"{func.__name__} completed",
                    component=component,
                    subcomponent=operation,
                    execution_time=f"{execution_time:.3f}s"
                )
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Create a default logger instance for easy import
# This ensures we have a single logger instance that can be imported directly
_configure_logging_once()
log = structlog.wrap_logger(logging.getLogger("app"))
log.setLevel(logging.INFO)


# ------------------------------------------------------------
# Helper for visibility banners in logs
# ------------------------------------------------------------


def banner(message: str):
    """Log a big banner for visibility in terminal logs."""
    logger = get_logger(__name__)
    line = "#" * 26
    banner_msg = f"\n{line}\n### {message} ###\n{line}\n"
    logger.info(banner_msg)