"""
vLLM-omni config utilities.

This module extends vLLM's config system to support diffusers format models
while maintaining compatibility with regular HuggingFace transformers models.
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional, Union
from transformers import PretrainedConfig
from huggingface_hub import try_to_load_from_cache


def _is_diffusers_format(model_path: Union[str, Path]) -> bool:
    """
    Checks if the given model path corresponds to a HuggingFace Diffusers format
    by looking for a 'model_index.json' file.
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)

    # Check locally first
    if os.path.isdir(model_path):
        return os.path.exists(os.path.join(model_path, "model_index.json"))
    
    # Try to check on HuggingFace Hub (simplified check)
    try:
        # This is a heuristic. A more robust check would involve listing repo files.
        # For now, we assume if it's not a local dir, it's a HF repo.
        # We can try to load model_index.json from cache or hub.
        cached_file = try_to_load_from_cache(
            repo_id=model_path,
            filename="model_index.json",
            revision=None,
        )
        return isinstance(cached_file, str)
    except Exception:
        # If we can't determine, assume it's not diffusers format
        return False


def _load_diffusers_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> PretrainedConfig:
    """
    Load configuration for a diffusers format model.
    
    Currently supports QwenImage models. Can be extended for other diffusers models.
    """
    if isinstance(model, Path):
        model = str(model)
    
    # For now, we assume all diffusers models are QwenImage
    # This can be extended to detect specific diffusers model types
    print(f"Loading diffusers format config from {model}")
    
    # Import here to avoid circular imports
    from .configs.qwen_image_config import QwenImageConfig
    
    # Load the diffusers config
    config = QwenImageConfig.from_pretrained(
        pretrained_model_name_or_path=model,
        revision=revision,
        **kwargs,
    )
    
    return config


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    config_format: str = "auto",
    hf_overrides_kw: Optional[dict[str, Any]] = None,
    hf_overrides_fn: Optional[Callable[[PretrainedConfig],
                                       PretrainedConfig]] = None,
    **kwargs,
) -> PretrainedConfig:
    """
    Extended get_config that supports both HuggingFace transformers and diffusers formats.
    
    This function first checks if the model is in diffusers format (by looking for
    model_index.json), and if so, loads the appropriate diffusers config.
    Otherwise, it falls back to vLLM's original get_config implementation.
    """
    
    # First check if this is a diffusers format model
    if _is_diffusers_format(model):
        print(f"Detected diffusers format model: {model}")
        try:
            return _load_diffusers_config(
                model=model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                hf_overrides_kw=hf_overrides_kw,
                hf_overrides_fn=hf_overrides_fn,
                **kwargs,
            )
        except Exception as e:
            print(f"Failed to load diffusers config for {model}: {e}")
            print("Falling back to HuggingFace transformers format")
    
    # Fall back to vLLM's original get_config for regular transformers models
    # Import here to avoid circular imports
    try:
        from vllm.transformers_utils.config import get_config as vllm_get_config
        return vllm_get_config(
            model=model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            config_format=config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
            **kwargs,
        )
    except ImportError:
        # If vLLM is not available, create a mock config
        print("vLLM not available, creating mock config")
        from transformers import PretrainedConfig
        return PretrainedConfig()


# Re-export the key functions
__all__ = [
    "get_config",
    "_is_diffusers_format", 
    "_load_diffusers_config",
]
