import torch
from typing import Dict, Any, Optional, Union, Iterable

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniDiTConfig, Qwen2_5OmniToken2WavDiTModel
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class QwenOmniDiTForDiffusionGeneration(nn.Module):
    """vLLM-compatible wrapper for transformers QwenOmni Token2Wav DiT model"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.model.",
        }
    )
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config
        self.prefix = prefix
        
        # TODO: Extract the DiT config from the main model config
        self.qwen2_5_omni_dit_config = self._extract_dit_config()
        
        # Initialize the DiT model using transformers implementation
        self.dit_model = Qwen2_5OmniToken2WavDiTModel(self.dit_config)
        
        # optional: Initialize the diffusion pipeline for orchestration
        # self._setup_for_diffusion_pipeline()
    def _extract_dit_config(self) -> Qwen2_5OmniDiTConfig:
        """Extract DiT configuration from the main model config"""
        hf_config = self.config.hf_config
        
        # Check if we have the token2wav_config in the HF config
        if hasattr(hf_config, 'token2wav_config') and hasattr(hf_config.token2wav_config, 'dit_config'):
            return hf_config.token2wav_config.dit_config
        
        # Fallback: create default config based on the actual HF config structure
        return Qwen2_5OmniDiTConfig(
            mel_dim=80,
            num_embeds=8193,
            emb_dim=512,
            hidden_size=1024,      # dim in HF config
            num_hidden_layers=22,  # depth in HF config
            num_attention_heads=16, # heads in HF config
            ff_mult=2,
            head_dim=64,
            repeats=2,
            block_size=1024,
            look_ahead_layers=[],
            look_backward_layers=[],
            dropout=0.1,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_vector: torch.Tensor,
        quantized_code: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Standard vLLM forward interface for single DiT step
        option 1: use kwargs to pass the input args
        option 2: re-design the input args, refer to transformers qwen omni token2wav model
        """
        # Call the DiT model for single step prediction
        noise_pred = self.dit_model(
            hidden_states=hidden_states,
            timestep=timestep,
            conditioning_vector=conditioning_vector,
            quantized_code=quantized_code,
            return_dict=False
        )[0]
        
        return noise_pred
    
    @torch.no_grad()
    def sample(
        self,
        conditioning_vector: torch.Tensor,
        reference_mel_spectrogram: torch.Tensor, # mel spectrogram latents
        quantized_code: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
        **kwargs
    ) -> torch.Tensor:
        """ Run the token2wav sampling pipeline
        Args:

        Returns:
            generated_mel_spectrogram: torch.Tensor, the generated mel spectrogram
        
        Notes: the args are the same as those of Qwen2_5OmniToken2WavDiTModel sample function in transformers
        """

        # option 1: re-write token2wav pipeline

        # option 2: re-use the token2wav sampling pipeline in transformers
        generated_mel_spectrogram = self.dit_model.sample(
            conditioning_vector=conditioning_vector,
            reference_mel_spectrogram=reference_mel_spectrogram,
            quantized_code=quantized_code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        return generated_mel_spectrogram
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with proper mapping for DiT model"""
        loader = AutoWeightsLoader(self.dit_model)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)