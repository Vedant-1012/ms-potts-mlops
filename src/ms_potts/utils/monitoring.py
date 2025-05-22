"""
Monitoring module for Ms. Potts MLOps project.

This module provides monitoring capabilities for tracking model performance,
system resource usage, and application health metrics.
"""

import time
import logging
import psutil
import threading
from datetime import datetime
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ms_potts_monitoring")

class ModelMonitor:
    """
    Monitor for tracking model performance metrics and system resources.
    """
    
    def __init__(self, metrics_dir="./metrics"):
        """
        Initialize the ModelMonitor.
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics = {
            "system": {},
            "model": {},
            "application": {}
        }
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.monitoring_thread = None
        # self.stop_monitoring = False
        self._stop_flag = False
        logger.info(f"ModelMonitor initialized. Metrics will be stored in {self.metrics_dir}")
    
    def start_monitoring(self, interval=5):
        """
        Start monitoring in a background thread.
        
        Args:
            interval: Time between monitoring checks in seconds
        """
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started system monitoring with interval of {interval} seconds")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("No monitoring thread is running")
            return
        
        self._stop_flag = True  # 🔄 Set flag
        # self.stop_monitoring = True
        self.monitoring_thread.join(timeout=10)
        logger.info("Stopped system monitoring")
    
    def _monitor_resources(self, interval):
        """
        Monitor system resources at regular intervals.
        
        Args:
            interval: Time between monitoring checks in seconds
        """
        while not self.stop_monitoring:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            timestamp = datetime.now().isoformat()
            
            system_metrics = {
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
            
            self.metrics["system"][timestamp] = system_metrics
            
            # Save metrics to file
            self._save_metrics("system", system_metrics)
            
            # Log summary
            logger.info(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
            
            # Wait for next interval
            time.sleep(interval)
    
    def log_model_metrics(self, metrics):
        """
        Log model performance metrics.
        
        Args:
            metrics: Dictionary of model metrics
        """
        timestamp = datetime.now().isoformat()
        metrics["timestamp"] = timestamp
        
        self.metrics["model"][timestamp] = metrics
        self._save_metrics("model", metrics)
        
        # Log summary
        metrics_summary = ", ".join([f"{k}: {v}" for k, v in metrics.items() if k != "timestamp"])
        logger.info(f"Model metrics - {metrics_summary}")
    
    def log_application_metrics(self, metrics):
        """
        Log application performance metrics.
        
        Args:
            metrics: Dictionary of application metrics
        """
        timestamp = datetime.now().isoformat()
        metrics["timestamp"] = timestamp
        
        self.metrics["application"][timestamp] = metrics
        self._save_metrics("application", metrics)
        
        # Log summary
        metrics_summary = ", ".join([f"{k}: {v}" for k, v in metrics.items() if k != "timestamp"])
        logger.info(f"Application metrics - {metrics_summary}")
    
    def _save_metrics(self, metric_type, metrics):
        """
        Save metrics to a JSON file.
        
        Args:
            metric_type: Type of metrics (system, model, application)
            metrics: Dictionary of metrics to save
        """
        filename = self.metrics_dir / f"{metric_type}_{metrics['timestamp'].split('T')[0]}.json"
        
        # Load existing metrics if file exists
        existing_metrics = []
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    existing_metrics = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error reading metrics file {filename}. Starting new file.")
        
        # Append new metrics
        existing_metrics.append(metrics)
        
        # Save updated metrics
        with open(filename, 'w') as f:
            json.dump(existing_metrics, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = ModelMonitor()
    
    # Start system monitoring
    monitor.start_monitoring(interval=2)
    
    # Simulate model metrics
    for i in range(5):
        # Simulate model inference
        time.sleep(1)
        
        # Log model metrics
        monitor.log_model_metrics({
            "accuracy": 0.92 + (i * 0.01),
            "loss": 0.15 - (i * 0.02),
            "latency_ms": 120 + (i * 5)
        })
        
        # Log application metrics
        monitor.log_application_metrics({
            "requests_per_second": 12 + i,
            "average_response_time_ms": 150 - (i * 10),
            "error_rate": 0.02 - (i * 0.002)
        })
    
    # Stop monitoring
    monitor.stop_monitoring()
