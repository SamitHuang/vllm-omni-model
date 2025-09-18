"""
vLLM-omni configuration module.

This module overrides vLLM's config to:
1. Use our extended get_config function that supports diffusers format models
2. Override ModelConfig to use our diffusion model registry when appropriate
"""

# Import everything from vLLM's config first
from vllm.config import *

# Import ModelConfig specifically and alias it for clarity
from vllm.config import ModelConfig as VllmModelConfig

# Import our extended config utilities
import vllm_omni.transformers_utils.config as config_utils
import vllm_omni.model_executor.models as me_models



class ModelConfig(VllmModelConfig):
    """
    Extended ModelConfig that uses our diffusion model registry.

    This class inherits from vLLM's ModelConfig but:
    1. Overrides __post_init__ to use our extended get_config function
    2. Overrides the registry property to use our ModelRegistryOmni
    """

    def __post_init__(self):
        """Override __post_init__ to use our extended get_config function."""
        # Call parent's __post_init__ but temporarily replace get_config
        import vllm.transformers_utils.config as vllm_config_utils

        # Save original get_config
        original_get_config = vllm_config_utils.get_config

        # Replace with our extended get_config
        vllm_config_utils.get_config = config_utils.get_config

        try:
            # Call parent's __post_init__ which will now use our get_config
            super().__post_init__()
        finally:
            # Restore original get_config
            vllm_config_utils.get_config = original_get_config

    @property
    def registry(self):
        """
        Return our ModelRegistryOmni for diffusion models.
        """
        return me_models.ModelRegistryOmni


# Re-export everything from vLLM's config
__all__ = [
    "VllmConfig",
    "ModelConfig",  # Our extended ModelConfig
    "CacheConfig",
    "DeviceConfig",
    "LoadConfig",
    "LoRAConfig",
    "MultiModalConfig",
    "ParallelConfig",
    "PromptAdapterConfig",
    "SchedulerConfig",
    "SpeculativeConfig",
    "TokenizerPoolConfig",
    # Add any other exports from vLLM's config as needed
]
