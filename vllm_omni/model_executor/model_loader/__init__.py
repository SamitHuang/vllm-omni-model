"""
vLLM-omni model loader.

We need to ensure get_model uses our ModelConfig (with overridden registry) instead of vLLM's.
"""

from typing import Optional
import torch.nn as nn

# Import our config classes which override me_models to use our registry
from vllm_omni.config import VllmConfig, ModelConfig

# Import vLLM's model loader utilities
from vllm.model_executor.model_loader import get_model_loader
from vllm.logger import init_logger

logger = init_logger(__name__)

def get_model(*,
              vllm_config: VllmConfig,
              model_config: Optional[ModelConfig] = None) -> nn.Module:
    """
    Load a diffusion model ensuring we use our ModelConfig with overridden registry.
    
    IMPORTANT: vllm_config must be created using vllm_omni.config.VllmConfig to ensure
    it contains our ModelConfig with the overridden registry.
    
    Args:
        vllm_config: VllmConfig from vllm_omni.config (NOT vllm.config)
        model_config: Optional ModelConfig override (from vllm_omni.config)
    """
    loader = get_model_loader(vllm_config.load_config)
    if model_config is None:
        # use model_config from vllm_omni instead of vllm so that it uses our registry diffusion models
        model_config = vllm_config.model_config
    
    return loader.load_model(vllm_config=vllm_config, model_config=model_config)


# Re-export vLLM's model loader utilities for convenience
from vllm.model_executor.model_loader import (
    get_model_loader, 
    register_model_loader,
    BaseModelLoader,
    DefaultModelLoader,
)

__all__ = [
    "get_model",              # vLLM's get_model (automatically uses our registry!)
    "get_model_loader",       # Re-export from vLLM
    "register_model_loader",  # Re-export from vLLM  
    "BaseModelLoader",        # Re-export from vLLM
    "DefaultModelLoader",     # Re-export from vLLM
    "VllmConfig",            # Our config classes
    "ModelConfig",
]
