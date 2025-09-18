"""
QwenImage configuration for diffusers format models.

This module defines a PretrainedConfig subclass that can load configurations
from diffusers format models (which have multiple config.json files in subfolders).
"""

import os
import json
from typing import Any, Dict, Optional
from transformers import PretrainedConfig, AutoConfig


class QwenImageConfig(PretrainedConfig):
    """
    Configuration class for QwenImage diffusers format models.
    
    This config aggregates configurations from multiple subfolders:
    - transformer/config.json
    - vae/config.json  
    - text_encoder/config.json
    - scheduler/scheduler_config.json
    """
    
    model_type = "qwen_image"
    
    def __init__(
        self,
        transformer_config: Optional[Dict[str, Any]] = None,
        vae_config: Optional[Dict[str, Any]] = None,
        text_encoder_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_config = transformer_config if transformer_config is not None else {}
        self.vae_config = vae_config if vae_config is not None else {}
        self.text_encoder_config = text_encoder_config if text_encoder_config is not None else {}
        self.scheduler_config = scheduler_config if scheduler_config is not None else {}

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        **kwargs
    ) -> "QwenImageConfig":
        """
        Loads the QwenImage configuration from a local path or HuggingFace Hub.
        
        This method aggregates configs from 'transformer', 'vae', 'text_encoder', 
        and 'scheduler' subfolders.
        """
        model_path = pretrained_model_name_or_path
        
        # Load transformer config
        transformer_config = cls._load_sub_config(model_path, "transformer", "config.json")
        
        # Load VAE config  
        vae_config = cls._load_sub_config(model_path, "vae", "config.json")
        
        # Load text encoder config (uses transformers.AutoConfig)
        text_encoder_config = cls._load_text_encoder_config(model_path)
        
        # Load scheduler config
        scheduler_config = cls._load_sub_config(model_path, "scheduler", "scheduler_config.json")
        
        return cls(
            transformer_config=transformer_config,
            vae_config=vae_config,
            text_encoder_config=text_encoder_config,
            scheduler_config=scheduler_config,
            **kwargs,
        )

    @staticmethod
    def _load_sub_config(
        model_path: str, 
        subfolder: str, 
        config_filename: str
    ) -> Dict[str, Any]:
        """Load a config file from a subfolder."""
        config_path = os.path.join(model_path, subfolder, config_filename)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

    @staticmethod
    def _load_text_encoder_config(model_path: str) -> Dict[str, Any]:
        """Load text encoder config using transformers.AutoConfig."""
        text_encoder_config_path = os.path.join(model_path, "text_encoder", "config.json")
        if os.path.exists(text_encoder_config_path):
            try:
                config = AutoConfig.from_pretrained(text_encoder_config_path)
                return config.to_dict()
            except Exception as e:
                print(f"Warning: Could not load text_encoder config from {text_encoder_config_path}: {e}")
        return {}
