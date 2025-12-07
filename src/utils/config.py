"""Configuration management"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration manager"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def database_url(self) -> str:
        return os.getenv("DATABASE_URL", "")

    @property
    def llm_model(self) -> str:
        return self._config.get("llm", {}).get("model", "gpt-4o")

    @property
    def batch_size(self) -> int:
        return self._config.get("performance", {}).get("batch_size", 20)

    @property
    def matching_weights(self) -> Dict[str, float]:
        return self._config.get("matching", {}).get("weights", {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

# Global config instance
config = Config()
