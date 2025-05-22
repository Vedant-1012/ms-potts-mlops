"""
Profiling utilities for Ms. Potts MLOps project.

This module provides profiling tools to analyze performance of Python code
and machine learning models.
"""

import cProfile
import pstats
import io
import time
import functools
import logging
import json
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("profiling.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ms_potts_profiling")

class CodeProfiler:
    """
    Utility for profiling Python code using cProfile.
    """
    
    def __init__(self, output_dir="./profiling_results"):
        """
        Initialize the CodeProfiler.
        
        Args:
            output_dir: Directory to store profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"CodeProfiler initialized. Results will be stored in {self.output_dir}")
    
    def profile_function(self, func):
        """
        Decorator to profile a function using cProfile.
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a profile object
            profiler = cProfile.Profile()
            
            # Start profiling
            profiler.enable()
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop profiling
                profiler.disable()
                
                # Generate a unique filename
                timestamp = int(time.time())
                filename = f"{func.__name__}_{timestamp}"
                
                # Save raw profiling results
                raw_path = self.output_dir / f"{filename}.prof"
                profiler.dump_stats(str(raw_path))
                
                # Save readable text results
                text_path = self.output_dir / f"{filename}.txt"
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                with open(text_path, 'w') as f:
                    f.write(s.getvalue())
                
                # Log summary
                logger.info(f"Profiled {func.__name__}. Results saved to {raw_path} and {text_path}")
        
        return wrapper
    
    def profile_block(self, name="code_block"):
        """
        Context manager to profile a block of code.
        
        Args:
            name: Name for the profiling results
        
        Returns:
            Context manager for profiling
        """
        class ProfilerContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.cprofile = cProfile.Profile()
            
            def __enter__(self):
                self.cprofile.enable()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.cprofile.disable()
                
                # Generate a unique filename
                timestamp = int(time.time())
                filename = f"{self.name}_{timestamp}"
                
                # Save raw profiling results
                raw_path = self.profiler.output_dir / f"{filename}.prof"
                self.cprofile.dump_stats(str(raw_path))
                
                # Save readable text results
                text_path = self.profiler.output_dir / f"{filename}.txt"
                s = io.StringIO()
                ps = pstats.Stats(self.cprofile, stream=s).sort_stats('cumulative')
                ps.print_stats()
                with open(text_path, 'w') as f:
                    f.write(s.getvalue())
                
                # Log summary
                logger.info(f"Profiled {self.name}. Results saved to {raw_path} and {text_path}")
        
        return ProfilerContext(self, name)

class MLProfiler:
    """
    Utility for profiling machine learning code, with special handling for PyTorch if available.
    """
    
    def __init__(self, output_dir="./ml_profiling_results"):
        """
        Initialize the MLProfiler.
        
        Args:
            output_dir: Directory to store profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if PyTorch is available
        self.has_torch = False
        self.has_torch_profiler = False
        
        try:
            import torch
            self.has_torch = True
            
            try:
                import torch.profiler
                self.has_torch_profiler = True
                logger.info("PyTorch profiler is available")
            except ImportError:
                logger.warning("PyTorch is available but profiler module is not")
        except ImportError:
            logger.warning("PyTorch is not available, using standard profiling only")
        
        logger.info(f"MLProfiler initialized. Results will be stored in {self.output_dir}")
    
    def profile_model_inference(self, model_func):
        """
        Decorator to profile model inference.
        
        Args:
            model_func: Function that performs model inference
        
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(model_func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()
            
            # Use PyTorch profiler if available
            if self.has_torch_profiler:
                import torch.profiler
                
                # Generate a unique filename
                timestamp = int(time.time())
                filename = f"{model_func.__name__}_{timestamp}"
                trace_path = self.output_dir / f"{filename}.json"
                
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    result = model_func(*args, **kwargs)
                
                # Save profiling results
                prof.export_chrome_trace(str(trace_path))
                
                # Generate text report
                text_path = self.output_dir / f"{filename}.txt"
                with open(text_path, 'w') as f:
                    f.write(str(prof.key_averages().table(sort_by="cpu_time_total")))
                
                logger.info(f"PyTorch profiling of {model_func.__name__} complete. Results saved to {trace_path} and {text_path}")
            else:
                # Fall back to standard profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = model_func(*args, **kwargs)
                finally:
                    profiler.disable()
                    
                    # Generate a unique filename
                    timestamp = int(time.time())
                    filename = f"{model_func.__name__}_{timestamp}"
                    
                    # Save raw profiling results
                    raw_path = self.output_dir / f"{filename}.prof"
                    profiler.dump_stats(str(raw_path))
                    
                    # Save readable text results
                    text_path = self.output_dir / f"{filename}.txt"
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats()
                    with open(text_path, 'w') as f:
                        f.write(s.getvalue())
                    
                    logger.info(f"Standard profiling of {model_func.__name__} complete. Results saved to {raw_path} and {text_path}")
            
            # Calculate and log execution time
            execution_time = time.time() - start_time
            logger.info(f"Execution time of {model_func.__name__}: {execution_time:.6f} seconds")
            
            return result
        
        return wrapper
    
    def profile_batch_processing(self, batch_size=1, iterations=10):
        """
        Decorator to profile batch processing performance.
        
        Args:
            batch_size: Size of each batch
            iterations: Number of iterations to profile
        
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Record performance metrics
                metrics = {
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "times": [],
                    "memory_usage": []
                }
                
                # Run iterations
                for i in range(iterations):
                    # Get memory usage before
                    if self.has_torch:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                    
                    # Time execution
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Record time
                    execution_time = end_time - start_time
                    metrics["times"].append(execution_time)
                    
                    # Record memory usage
                    if self.has_torch:
                        import torch
                        if torch.cuda.is_available():
                            mem_after = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                            metrics["memory_usage"].append(mem_after - mem_before)
                    
                    logger.info(f"Iteration {i+1}/{iterations}: {execution_time:.6f} seconds")
                
                # Calculate statistics
                metrics["avg_time"] = sum(metrics["times"]) / len(metrics["times"])
                metrics["min_time"] = min(metrics["times"])
                metrics["max_time"] = max(metrics["times"])
                
                if metrics["memory_usage"]:
                    metrics["avg_memory"] = sum(metrics["memory_usage"]) / len(metrics["memory_usage"])
                    metrics["max_memory"] = max(metrics["memory_usage"])
                
                # Save metrics
                timestamp = int(time.time())
                metrics_path = self.output_dir / f"{func.__name__}_batch_metrics_{timestamp}.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Log summary
                logger.info(f"Batch profiling complete for {func.__name__}")
                logger.info(f"Average execution time: {metrics['avg_time']:.6f} seconds")
                if metrics["memory_usage"]:
                    logger.info(f"Average memory usage: {metrics['avg_memory']:.2f} MB")
                logger.info(f"Detailed metrics saved to {metrics_path}")
                
                return result
            
            return wrapper
        
        return decorator

# Example usage
if __name__ == "__main__":
    # Create profilers
    code_profiler = CodeProfiler()
    ml_profiler = MLProfiler()
    
    # Example function to profile
    @code_profiler.profile_function
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # Profile a code block
    with code_profiler.profile_block("example_block"):
        result = fibonacci(20)
        logger.info(f"Fibonacci result: {result}")
    
    # Example ML function (if PyTorch is available)
    if ml_profiler.has_torch:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        
        @ml_profiler.profile_model_inference
        def predict(input_tensor):
            return model(input_tensor)
        
        # Profile model inference
        input_data = torch.randn(32, 10)
        output = predict(input_data)
        logger.info(f"Model output shape: {output.shape}")
        
        # Profile batch processing
        @ml_profiler.profile_batch_processing(batch_size=32, iterations=5)
        def process_batch(batch_size):
            data = torch.randn(batch_size, 10)
            return model(data)
        
        process_batch(32)
