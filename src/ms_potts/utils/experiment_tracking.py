"""
Experiment tracking module for Ms. Potts MLOps project.

This module provides experiment tracking capabilities using MLflow,
allowing for tracking of parameters, metrics, artifacts, and models.
"""

import os
import logging
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_tracking.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ms_potts_experiment_tracking")

class ExperimentTracker:
    """
    Experiment tracking utility using MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str = "ms_potts_experiments",
        tracking_uri: Optional[str] = None,
        artifacts_dir: str = "./mlruns",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the ExperimentTracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (local or remote)
            artifacts_dir: Directory to store artifacts
            tags: Tags to apply to the experiment
        """
        self.experiment_name = experiment_name
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory for tracking
            local_tracking_uri = f"file://{os.path.abspath(str(self.artifacts_dir))}"
            mlflow.set_tracking_uri(local_tracking_uri)
        
        # Set or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags=tags or {}
            )
            logger.info(f"Created new experiment '{experiment_name}' with ID: {self.experiment_id}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {self.experiment_id}")
        
        mlflow.set_experiment(experiment_name)
        self.active_run = None
        
        logger.info(f"ExperimentTracker initialized with tracking URI: {mlflow.get_tracking_uri()}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Start a new tracking run.
        
        Args:
            run_name: Name for the run
            tags: Tags to apply to the run
            description: Description of the run
        
        Returns:
            MLflow run object
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        # Set up tags
        run_tags = tags or {}
        if description:
            run_tags["description"] = description
        
        # Start the run
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=run_tags
        )
        
        logger.info(f"Started run '{run_name}' with ID: {self.active_run.info.run_id}")
        return self.active_run
    
    def end_run(self):
        """End the current tracking run."""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended run with ID: {self.active_run.info.run_id}")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        # Convert non-string values to strings
        for key, value in params.items():
            if not isinstance(value, (str, int, float, bool)):
                params[key] = str(value)
        
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step or iteration number (optional)
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        mlflow.log_metrics(metrics, step=step)
        
        # Log summary for important metrics
        metrics_summary = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_info = f" at step {step}" if step is not None else ""
        logger.info(f"Logged metrics{step_info}: {metrics_summary}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact for the current run.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Path within the artifact directory (optional)
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all artifacts in a directory for the current run.
        
        Args:
            local_dir: Path to the local directory to log
            artifact_path: Path within the artifact directory (optional)
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.info(f"Logged artifacts from directory: {local_dir}")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        conda_env: Optional[Union[Dict[str, Any], str]] = None,
        code_paths: Optional[List[str]] = None,
        registered_model_name: Optional[str] = None
    ):
        """
        Log a model for the current run.
        
        Args:
            model: Model object to log
            artifact_path: Path within the artifact directory
            conda_env: Conda environment for the model
            code_paths: Paths to code files to log with the model
            registered_model_name: Name to register the model with in the registry
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        # Determine the type of model and use appropriate logging function
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                    code_paths=code_paths,
                    registered_model_name=registered_model_name
                )
                logger.info(f"Logged PyTorch model to {artifact_path}")
                return
        except ImportError:
            pass
        
        # Default to generic model logging
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            conda_env=conda_env,
            code_paths=code_paths,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged generic model to {artifact_path}")
    
    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib or plotly figure for the current run.
        
        Args:
            figure: Figure object to log
            artifact_file: Filename for the artifact
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        # Create a temporary file for the figure
        temp_dir = Path("./temp_figures")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / artifact_file
        
        # Save the figure based on its type
        try:
            # Try matplotlib
            figure.savefig(temp_path)
            mlflow.log_artifact(str(temp_path), "figures")
            logger.info(f"Logged matplotlib figure to figures/{artifact_file}")
            return
        except (AttributeError, TypeError):
            pass
        
        try:
            # Try plotly
            import plotly
            if isinstance(figure, plotly.graph_objs.Figure):
                figure.write_image(str(temp_path))
                mlflow.log_artifact(str(temp_path), "figures")
                logger.info(f"Logged plotly figure to figures/{artifact_file}")
                return
        except (ImportError, AttributeError):
            pass
        
        logger.warning(f"Could not log figure. Unsupported figure type.")
    
    def get_run_url(self):
        """
        Get the URL for the current run.
        
        Returns:
            URL string for the current run
        """
        if not self.active_run:
            logger.warning("No active run.")
            return None
        
        tracking_uri = mlflow.get_tracking_uri()
        run_id = self.active_run.info.run_id
        
        # Handle different tracking URI types
        if tracking_uri.startswith("http"):
            # Remote tracking server
            return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{run_id}"
        else:
            # Local tracking
            return f"Local run ID: {run_id} (use mlflow ui to view)"
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ):
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter string for search
            max_results: Maximum number of results to return
            order_by: List of columns to order by
        
        Returns:
            List of run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["metrics.accuracy DESC"]
        )
        
        logger.info(f"Found {len(runs)} runs matching filter: {filter_string}")
        return runs

# Example usage
if __name__ == "__main__":
    # Create experiment tracker
    tracker = ExperimentTracker(experiment_name="example_experiment")
    
    # Start a run
    with tracker.start_run(run_name="example_run", description="Example run for demonstration"):
        # Log parameters
        tracker.log_params({
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "Adam"
        })
        
        # Simulate training loop
        for epoch in range(10):
            # Simulate metrics
            metrics = {
                "accuracy": 0.75 + epoch * 0.02,
                "loss": 0.5 - epoch * 0.03,
                "val_accuracy": 0.70 + epoch * 0.02,
                "val_loss": 0.55 - epoch * 0.03
            }
            
            # Log metrics
            tracker.log_metrics(metrics, step=epoch)
        
        # Create and log a sample artifact
        with open("sample_output.json", "w") as f:
            json.dump({"prediction": [0.1, 0.2, 0.7]}, f)
        
        tracker.log_artifact("sample_output.json")
        
        # Log run URL
        print(f"Run URL: {tracker.get_run_url()}")
    
    # Search for runs
    runs = tracker.search_runs(filter_string="metrics.accuracy > 0.8")
    print(f"Found {len(runs)} runs with accuracy > 0.8")
