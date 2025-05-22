"""
Enhanced logging module for Ms. Potts MLOps project.

This module provides comprehensive logging capabilities using Python's
logging module with rich formatting and output options.
"""

import logging
import sys
import os
from pathlib import Path
import json
import datetime
from typing import Optional, Dict, Any, Union

# Import rich for enhanced console output
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.traceback import install as install_rich_traceback
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

class EnhancedLogger:
    """
    Enhanced logging utility with rich formatting and multiple outputs.
    """
    
    def __init__(
        self,
        name: str = "ms_potts",
        level: str = "info",
        log_dir: str = "./logs",
        console_output: bool = True,
        file_output: bool = True,
        json_output: bool = False,
        rich_formatting: Optional[bool] = None
    ):
        """
        Initialize the EnhancedLogger.
        
        Args:
            name: Logger name
            level: Log level (debug, info, warning, error, critical)
            log_dir: Directory to store log files
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            json_output: Whether to output logs in JSON format
            rich_formatting: Whether to use rich formatting (if available)
        """
        self.name = name
        self.level = LOG_LEVELS.get(level.lower(), logging.INFO)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine whether to use rich formatting
        if rich_formatting is None:
            self.rich_formatting = RICH_AVAILABLE
        else:
            self.rich_formatting = rich_formatting and RICH_AVAILABLE
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Set up console output
        if console_output:
            self._setup_console_handler()
        
        # Set up file output
        if file_output:
            self._setup_file_handler(json_format=json_output)
        
        # Install rich traceback if available and enabled
        if self.rich_formatting:
            install_rich_traceback()
        
        self.logger.info(f"Logger '{name}' initialized with level {level}")
    
    def _setup_console_handler(self):
        """Set up console handler with appropriate formatting."""
        if self.rich_formatting:
            # Use rich handler for enhanced console output
            console = Console()
            handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                omit_repeated_times=False
            )
            handler.setLevel(self.level)
            self.logger.addHandler(handler)
        else:
            # Use standard console handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_file_handler(self, json_format=False):
        """
        Set up file handler with appropriate formatting.
        
        Args:
            json_format: Whether to use JSON formatting
        """
        # Create log filename with date
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"{self.name}_{date_str}.log"
        
        # Create handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(self.level)
        
        if json_format:
            # Use JSON formatter
            formatter = JsonFormatter()
        else:
            # Use standard formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger
    
    def set_level(self, level: str):
        """
        Set the log level.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
        """
        log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)
        self.logger.info(f"Log level set to {level}")

class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format.
    """
    
    def format(self, record):
        """
        Format the log record as JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted JSON string
        """
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_data.update(record.data)
        
        return json.dumps(log_data)

def log_with_context(logger, level, message, **context):
    """
    Log a message with additional context data.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Additional context data to include in the log
    """
    # Get the log method based on level
    log_method = getattr(logger, level.lower(), logger.info)
    
    # Create a log record with extra data
    extra = {"data": context}
    log_method(message, extra=extra)

# Example usage
if __name__ == "__main__":
    # Create enhanced logger
    enhanced_logger = EnhancedLogger(
        name="example_logger",
        level="debug",
        console_output=True,
        file_output=True,
        json_output=True
    )
    
    logger = enhanced_logger.get_logger()
    
    # Log messages at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log with context
    log_with_context(
        logger,
        "info",
        "Processing user request",
        user_id=123,
        request_path="/api/data",
        processing_time=0.05
    )
    
    # Log an exception
    try:
        result = 1 / 0
    except Exception as e:
        logger.exception("An error occurred during calculation")
