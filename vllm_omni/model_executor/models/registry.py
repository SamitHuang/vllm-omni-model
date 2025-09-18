"""
Registry for vLLM-omni diffusion models.

This module creates a separate ModelRegistryOmni for diffusion models to avoid
conflicts with vLLM's registry.

Conflict example:
- vLLM maps: Qwen2_5OmniModel -> Qwen2_5OmniThinkerForConditionalGeneration (AR text generation)
- vLLM-omni maps: Qwen2_5OmniModel -> QwenOmniDiTForDiffusionGeneration (diffusion only)

Solution: Create separate ModelRegistryOmni that doesn't interfere with vLLM's registry.
"""

from typing import Optional, Dict, Any, Type
import logging

# Simple logger for when vLLM is not available
logger = logging.getLogger(__name__)

# Simple registry implementation that doesn't depend on vLLM
class _LazyRegisteredModel:
    """Simple lazy model registration."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._model_cls = None

    def load_model_cls(self):
        """Load the model class lazily."""
        if self._model_cls is None:
            try:
                module = __import__(self.module_name, fromlist=[self.class_name])
                self._model_cls = getattr(module, self.class_name)
            except ImportError as e:
                logger.warning(f"Could not import {self.module_name}.{self.class_name}: {e}")
                # Create a mock class for testing
                class MockModel:
                    def __init__(self, *args, **kwargs):
                        pass
                self._model_cls = MockModel
        return self._model_cls


class _ModelRegistry:
    """Simple model registry that doesn't depend on vLLM."""

    def __init__(self, models: Dict[str, _LazyRegisteredModel]):
        self.models = models

    def load_model_cls(self, model_arch: str):
        """Load a model class by architecture name."""
        if model_arch not in self.models:
            raise ValueError(f"Unknown model architecture: {model_arch}")
        return self.models[model_arch].load_model_cls()

    def get_supported_archs(self):
        """Get all supported architectures."""
        return list(self.models.keys())

# NOTE： vllm registry map Qwen2_5OmniModel to Qwen2_5OmniThinkerForConditionalGeneration
# but in vllm-omni, we need to map Qwen2_5OmniModel to QwenOmniDiTForDiffusionGeneration, say ModelRegistryOmni
# it means we need to create a new ModelRegistry for vllm-omni.
'''
_MULTIMODAL_MODELS = {
    "Qwen2_5OmniModel": ("qwen2_5_omni_thinker", "Qwen2_5OmniThinkerForConditionalGeneration"),  # noqa: E501
    "Qwen2_5OmniForConditionalGeneration": ("qwen2_5_omni_thinker", "Qwen2_5OmniThinkerForConditionalGeneration"),  # noqa: E501
}
'''

# Define diffusion models supported by vLLM-omni
_DIFFUSION_MODELS = {
    # QwenOmni models (transformers-style)
    # Main architecture from HF config - this conflicts with vLLM's mapping
    "Qwen2_5OmniModel": ("qwen2_5_omni", "QwenOmniDiTForDiffusionGeneration"),

    # QwenImage models (diffusers-style)
    "QwenImageTransformer2DModel": ("qwen_image", "QwenImageDiTForDiffusionGeneration"),
    "QwenImagePipeline": ("qwen_image", "QwenImageDiTForDiffusionGeneration"),
}

# Create separate ModelRegistryOmni for diffusion models (completely independent from vLLM)
ModelRegistryOmni = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=f"vllm_omni.model_executor.models.{mod_relname}",
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _DIFFUSION_MODELS.items()
})

def is_registered_diffusion_model(model_arch: str) -> bool:
    """Check if a model architecture is registered in our diffusion registry"""
    return model_arch in _DIFFUSION_MODELS

def get_diffusion_models():
    """Get all registered diffusion models"""
    return _DIFFUSION_MODELS.copy()

def get_diffusion_model_cls(model_arch: str):
    """Load a diffusion model class from our separate registry"""
    if not is_registered_diffusion_model(model_arch):
        raise ValueError(f"'{model_arch}' is not a registered diffusion model")
    return ModelRegistryOmni.load_model_cls(model_arch)

def get_supported_diffusion_archs():
    """Get all supported diffusion model architectures"""
    return ModelRegistryOmni.get_supported_archs()

# Export our separate registry (NOT vLLM's registry)
ModelRegistry = ModelRegistryOmni
