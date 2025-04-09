"""
Configuration loading utilities for the Smite 2 Translation Agent.

Provides functionality to load and access configuration from the central config.py.
"""

import os
import json
import logging
from typing import Dict, Any

# Import the central configuration
from smite2_translation.config import (
    DEFAULT_MODEL,
    RULESET_DIR,
    SUPPORTED_LANGUAGES,
    MAX_RETRIES,
    RETRY_DELAY
)

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration, with the option to override with a JSON file.
    
    Args:
        config_path: Optional path to a JSON configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    # Start with the configuration from config.py
    config = {
        "model": DEFAULT_MODEL,
        "ruleset_directory": RULESET_DIR,
        "supported_languages": SUPPORTED_LANGUAGES,
        "retry_attempts": MAX_RETRIES,
        "retry_delay": RETRY_DELAY
    }
    
    # Try to load from file if path specified (for overrides)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration overrides from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_path}: {str(e)}")
            logger.warning("Using default configuration values")
    
    return config

def get_default_model() -> str:
    """Get the default model name from configuration."""
    return DEFAULT_MODEL

# Example of how it might evolve to load from a file:
# import yaml
# def load_config_from_file(filepath="config.yaml") -> Dict[str, Any]:
#     try:
#         with open(filepath, 'r') as f:
#             config = yaml.safe_load(f)
#         # Merge with environment variables (env overrides file)
#         env_api_key = os.environ.get('OPENAI_API_KEY')
#         if env_api_key:
#             config['openai_api_key'] = env_api_key
#         return config or {}
#     except FileNotFoundError:
#         logger.warning(f"Config file not found: {filepath}. Using defaults/env vars.")
#         # Fallback to basic load
#         return load_config()
#     except Exception as e:
#         logger.error(f"Error loading config file {filepath}: {e}", exc_info=True)
#         # Fallback to basic load
#         return load_config() 