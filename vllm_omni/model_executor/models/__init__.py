"""
vLLM-omni model executor models.

This module provides the registry and interfaces for diffusion models in vLLM-omni.
Specific model classes are loaded on-demand via the registry.
"""

from .registry import (
    ModelRegistry,              # Our separate ModelRegistryOmni
    ModelRegistryOmni,          # Explicit export
    get_diffusion_models,       # List available diffusion models
    get_diffusion_model_cls,    # Load specific diffusion model class
    get_supported_diffusion_archs,  # Get supported architectures
)

__all__ = [
    "ModelRegistry",            # Our separate registry for diffusion models
    "ModelRegistryOmni",        # Explicit name
    "get_diffusion_models",     # Get list of diffusion models
    "get_diffusion_model_cls",  # Load diffusion model class
    "get_supported_diffusion_archs",  # Get supported architectures
]
