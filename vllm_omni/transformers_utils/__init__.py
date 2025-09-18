"""
vLLM-omni transformers utilities.

This module provides extended transformers utilities that support both
HuggingFace transformers and diffusers format models.
"""

from .config import get_config

__all__ = [
    "get_config",
]
