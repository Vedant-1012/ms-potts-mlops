"""
Configuration management module for Ms. Potts MLOps project.

This module provides configuration management capabilities using Hydra,
allowing for hierarchical configuration with command-line overrides.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("config_management.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ms_potts_config_management")

# Define configuration dataclasses
@dataclass
class ModelConfig:
    """Configuration for the model."""
    name: str = "sentence-transformers"
    embedding_size: int = 768
    model_path: str = "all-MiniLM-L6-v2"
    use_gpu: bool = True
    batch_size: int = 32

@dataclass
class RetrieverConfig:
    """Configuration for the retriever component."""
    index_path: str = "data/faiss_index"
    chunk_size: int = 512
    overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.6

@dataclass
class GeminiConfig:
    """Configuration for the Gemini API integration."""
    api_key: str = "${oc.env:GEMINI_API_KEY,}"
    model_name: str = "gemini-pro"
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40

@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    interim_data_path: str = "data/interim"
    external_data_path: str = "data/external"
    nutrition_data_file: str = "nutrition_data.json"
    intent_examples_file: str = "intent_examples.json"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "Adam"
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    random_seed: int = 42

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "info"
    log_dir: str = "logs"
    console_output: bool = True
    file_output: bool = True
    json_output: bool = False
    rich_formatting: bool = True

@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    enabled: bool = True
    metrics_dir: str = "metrics"
    interval: int = 5
    save_system_metrics: bool = True
    save_model_metrics: bool = True
    save_application_metrics: bool = True

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    tracking_uri: Optional[str] = None
    experiment_name: str = "ms_potts_experiments"
    artifacts_dir: str = "mlruns"
    log_system_info: bool = True

@dataclass
class AppConfig:
    """Configuration for the application."""
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    workers: int = 1
    timeout: int = 60

@dataclass
class MsPottsConfig:
    """Main configuration for Ms. Potts."""
    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    app: AppConfig = field(default_factory=AppConfig)
    mode: str = "production"  # Options: development, production, testing

# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="ms_potts_config", node=MsPottsConfig)

class ConfigurationManager:
    """
    Configuration management utility using Hydra.
    """
    
    def __init__(self, config_dir: str = "conf"):
        """
        Initialize the ConfigurationManager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Create default configuration files if they don't exist
        self._create_default_configs()
        
        logger.info(f"ConfigurationManager initialized with config directory: {self.config_dir}")
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist."""
        # Create main config.yaml
        main_config_path = self.config_dir / "config.yaml"
        if not main_config_path.exists():
            default_config = MsPottsConfig()
            with open(main_config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(default_config))
            logger.info(f"Created default config.yaml at {main_config_path}")
        
        # Create environment-specific configs
        for env in ["development", "production", "testing"]:
            env_dir = self.config_dir / env
            env_dir.mkdir(exist_ok=True)
            
            env_config_path = env_dir / "config.yaml"
            if not env_config_path.exists():
                # Create environment-specific overrides
                env_config = {
                    "mode": env,
                    "logging": {"level": "debug" if env == "development" else "info"},
                    "app": {"debug": env == "development"}
                }
                
                with open(env_config_path, 'w') as f:
                    f.write(OmegaConf.to_yaml(env_config))
                logger.info(f"Created {env} config at {env_config_path}")
    
    def get_config_path(self):
        """Get the path to the configuration directory."""
        return str(self.config_dir)

# Hydra entry point for the application
@hydra.main(config_path="conf", config_name="config")
def run_app(cfg: DictConfig) -> None:
    """
    Run the application with the given configuration.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info(f"Running with configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Access configuration values
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Mode: {cfg.mode}")
    
    # Example of using the configuration
    if cfg.mode == "development":
        logger.info("Running in development mode")
    elif cfg.mode == "production":
        logger.info("Running in production mode")
    elif cfg.mode == "testing":
        logger.info("Running in testing mode")
    
    # Here you would initialize and run your application
    # using the configuration values
    
    return cfg

# Example usage
if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # This will be called by Hydra
    run_app()
