"""
vLLM-omni transformers utils configs.

This module registers custom config classes for diffusers format models.
"""

from .qwen_image_config import QwenImageConfig

__all__ = [
    "QwenImageConfig",
]
