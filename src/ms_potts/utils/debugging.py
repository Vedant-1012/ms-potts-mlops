"""
Debugging utilities for Ms. Potts MLOps project.

This module provides debugging tools and utilities to help identify and fix issues
in the ML pipeline and application.
"""

import sys
import traceback
import inspect
import logging
import time
import functools
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ms_potts_debugging")

class DebugTracer:
    """
    Utility for tracing function calls and execution flow.
    """
    
    def __init__(self, output_dir="./debug_traces"):
        """
        Initialize the DebugTracer.
        
        Args:
            output_dir: Directory to store debug traces
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.trace_id = int(time.time())
        self.trace_data = []
        logger.info(f"DebugTracer initialized. Traces will be stored in {self.output_dir}")
    
    def trace_function(self, func):
        """
        Decorator to trace function calls, arguments, and return values.
        
        Args:
            func: Function to trace
        
        Returns:
            Wrapped function with tracing
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Record function entry
            entry_time = time.time()
            frame = inspect.currentframe()
            filename = frame.f_back.f_code.co_filename
            line_number = frame.f_back.f_lineno
            
            # Format arguments for logging (handle non-serializable objects)
            safe_args = []
            for arg in args:
                try:
                    json.dumps(arg)
                    safe_args.append(arg)
                except (TypeError, OverflowError):
                    safe_args.append(str(arg))
            
            safe_kwargs = {}
            for k, v in kwargs.items():
                try:
                    json.dumps(v)
                    safe_kwargs[k] = v
                except (TypeError, OverflowError):
                    safe_kwargs[k] = str(v)
            
            entry_data = {
                "event": "function_entry",
                "timestamp": entry_time,
                "function": func.__name__,
                "module": func.__module__,
                "filename": filename,
                "line": line_number,
                "args": safe_args,
                "kwargs": safe_kwargs
            }
            
            self.trace_data.append(entry_data)
            logger.debug(f"TRACE: Entering {func.__name__} at {filename}:{line_number}")
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Record function exit
                exit_time = time.time()
                
                # Format result for logging (handle non-serializable objects)
                safe_result = None
                try:
                    json.dumps(result)
                    safe_result = result
                except (TypeError, OverflowError):
                    safe_result = str(result)
                
                exit_data = {
                    "event": "function_exit",
                    "timestamp": exit_time,
                    "function": func.__name__,
                    "duration": exit_time - entry_time,
                    "result": safe_result,
                    "success": True
                }
                
                self.trace_data.append(exit_data)
                logger.debug(f"TRACE: Exiting {func.__name__}, duration: {exit_time - entry_time:.6f}s")
                
                return result
            
            except Exception as e:
                # Record function exception
                exit_time = time.time()
                exception_data = {
                    "event": "function_exception",
                    "timestamp": exit_time,
                    "function": func.__name__,
                    "duration": exit_time - entry_time,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc(),
                    "success": False
                }
                
                self.trace_data.append(exception_data)
                logger.error(f"TRACE: Exception in {func.__name__}: {str(e)}")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    def save_trace(self, filename=None):
        """
        Save the current trace data to a file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
        
        Returns:
            Path to the saved trace file
        """
        if filename is None:
            filename = f"trace_{self.trace_id}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.trace_data, f, indent=2)
        
        logger.info(f"Saved trace data to {filepath}")
        return filepath
    
    def clear_trace(self):
        """Clear the current trace data."""
        self.trace_data = []
        self.trace_id = int(time.time())
        logger.info("Cleared trace data")

def debug_value(value, name="Value"):
    """
    Debug utility to inspect a value and its type.
    
    Args:
        value: Value to inspect
        name: Name to use in the debug output
    """
    value_type = type(value).__name__
    
    if hasattr(value, 'shape'):  # For numpy arrays, tensors
        shape_info = f", shape: {value.shape}"
    elif hasattr(value, '__len__'):  # For lists, dicts, etc.
        shape_info = f", length: {len(value)}"
    else:
        shape_info = ""
    
    logger.debug(f"DEBUG: {name} (type: {value_type}{shape_info}) = {value}")
    return value

def exception_handler(func):
    """
    Decorator to catch and log exceptions.
    
    Args:
        func: Function to wrap with exception handling
    
    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    return wrapper

# Example usage
if __name__ == "__main__":
    # Create tracer
    tracer = DebugTracer()
    
    # Example function to trace
    @tracer.trace_function
    def example_function(a, b):
        logger.info(f"Running example function with {a} and {b}")
        result = a + b
        debug_value(result, "Addition result")
        return result
    
    # Run the function
    try:
        result = example_function(5, 7)
        logger.info(f"Function returned: {result}")
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
    
    # Save the trace
    tracer.save_trace("example_trace.json")
