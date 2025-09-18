from typing import Dict, Any, Optional, Union, Iterable, List
import torch
import torch.nn as nn
from diffusers import QwenImageTransformer2DModel
from diffusers import QwenImagePipeline, QwenImageEditPipeline
from torchvision.io import image
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class QwenImageForDiffusionGeneration(nn.Module):
    """vLLM-compatible wrapper for diffusersQwenImagePipeline """
    # TODO: check weight name mapping
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
        }
    )
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Extract model config from vllm config
        self.config = vllm_config.model_config
        self.prefix = prefix
        self.dit_config= self._extract_dit_config()

        # Initialize the qwen image dit model
        self.dit_model = QwenImageTransformer2DModel(**self.dit_config)

        # setup diffusion pipeline
        self._setup_for_diffusion_pipeline()
    
    def _setup_for_diffusion_pipeline(self):
        """prepare all necessary components for pipeline running, such as timestep scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler 
        # create diffusion scheduler
        self.scheduler_config = self.model_config.scheduler_config
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config) 
        # create diffusion pipeline
        self.pipeline_name = self.config._class_name
        self.pipeline = QwenImagePipeline(
            scheduler = self.scheduler,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=self.dit_model,
        )
        # enable optimization from diffusers pipeline
        self.pipeline.enable_xformers_memory_efficient_attention()
    def _extract_dit_config(self) -> dict:
        """Extract DiT configuration from the main model config"""
        if self.config.hasattr('transformer_config'):
            # https://huggingface.co/Qwen/Qwen-Image/blob/main/transformer/config.json
            return self.config.transformer_config
        # Fallback: create default config
        return dict(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=3,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),
            # ... other required parameters
        )
    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, conditions: dict, **kwargs):
        ''' Dit model one step forward
        This is not used for models from diffusers. TODO: nn.Module
        '''
        latents = self.dit_model(
            hidden_states=hidden_states,
            timestep=timestep,
            conditions=conditions,
            return_dict=False
        )[0]

        return latents
    
    @torch.no_grad()
    def sample(self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        """ Perform diffusion pipeline running. Re-use the pipeline from diffusers
        Args:

        Returns:
            latents: torch.Tensor, the generated latents

        Notes: changes from pipeline input args:
            1) require prompt_embeds, which is generated from Text Encoder run on vllm engine core
            2) remove return_type, which is forced to 'latents'. the return latents will be processed by vae decoding in AsynLLM outut processor
            3) remove unused args: callback_on_step_end
        """
        latents = self.pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_type='latents',
            generator=generator,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            sigmas=sigmas,
            latents=latents,
        )
        
        return latents
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # TODO: check whether diffusers weight loading is correct
        loader = AutoWeightsLoader(self.dit_model)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

# NOTE: rewrite _setup_for_diffusion_pipeline for sample function for QwenImageEditPipeline
class QwenImageEditForDiffusionGeneration(QwenImageForDiffusionGeneration):
    
    def _setup_for_diffusion_pipeline(self):
        """prepare all necessary components for pipeline running, such as timestep scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler 
        # create diffusion scheduler
        self.scheduler_config = self.model_config.scheduler_config
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config) 
        # create diffusion pipeline
        self.pipeline_name = self.config._class_name
        self.pipeline = QwenImageEditPipeline(
            scheduler = self.scheduler,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=self.dit_model,
        )
        # enable optimization from diffusers pipeline
        self.pipeline.enable_xformers_memory_efficient_attention()

    @torch.no_grad()
    def sample(self,
        image_embeds: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        """ Perform diffusion pipeline running. Re-use the pipeline from diffusers"""
        latents = self.pipeline(
            image_embeds=image_embeds,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_type='latents',
            generator=generator,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            sigmas=sigmas,
            latents=latents,
        )
        
        return latents
 


'''
# option 2: all pipelines in one class
class QwenImageForDiffusion(nn.Module):
    # TODO: check weight name mapping
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
        }
    )
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Extract model config from vllm config
        self.config = vllm_config.model_config
        self.prefix = prefix

        self.dit_config= self._extract_dit_config()

        # Initialize the qwen imagemodel
        self.dit_model = QwenImageTransformer2DModel(**self.dit_config)

        # Identify the pipeline type from the config
        self._setup_for_diffusion_pipeline()
    
    def _setup_for_diffusion_pipeline(self):
        """prepare all necessary components for pipeline running, such as timestep scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler 
        self.scheduler_config = self.model_config.scheduler_config
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config) 
        self.pipeline_name = self.config._class_name

        if self.pipeline_name == 'QwenImagePipeline':
            from diffusers import QwenImagePipeline
            self.pipeline = QwenImagePipeline(
                scheduler = self.scheduler,
                vae=None,
                text_encoder=None,
                tokenizer=None,
                transformer=self.dit_model,
            )
            self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color..." 
        if self.pipeline_name == 'QwenImageEditPipeline':
            from diffusers import QwenImageEditPipeline
            self.pipeline = QwenImageEditPipeline(
                scheduler = self.scheduler,
                vae=None,
                text_encoder=None,
                tokenizer=None,
                transformer=self.dit_model,
            )
            self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color... " 
        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_name}")
    def _extract_dit_config(self) -> dict:
        """Extract DiT configuration from the main model config"""
        # The config should contain dit_config section
        # Fallback: create default config
        return dict(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=3,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),
            # ... other required parameters
        )
    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, conditions: dict, **kwargs):
        pass
    
    def sample(self, conditions: dict, num_inference_steps: int, **kwargs):
        """ Perform diffusion pipeline running
        Args:
        main inference interface for the model
        """
        if self.pipeline_name == 'QwenImagePipeline':
            latents = self.pipeline(
                prompt_embeds=conditions['prompt_embeds'],
                prompt_embeds_mask=conditions['prompt_embeds_mask'],
                negative_prompt_embeds=conditions['negative_prompt_embeds'],
                negative_prompt_embeds_mask=conditions['negative_prompt_embeds_mask'],
                true_cfg_scale=conditions['true_cfg_scale'],
                height=conditions['height'],
                width=conditions['width'],
                num_inference_steps=num_inference_steps,
                guidance_scale=conditions['guidance_scale'],
                return_type='latents',
            )
        elif self.pipeline_name == 'QwenImageEditPipeline':
            # FIXME: we need to change QwenImageEditPipeline to accept image_embeds input
            latents = self.pipeline(
                image_embeds=conditions['image_embeds'],
                prompt_embeds=conditions['prompt_embeds'],
                prompt_embeds_mask=conditions['prompt_embeds_mask'],
                negative_prompt_embeds=conditions['negative_prompt_embeds'],
                negative_prompt_embeds_mask=conditions['negative_prompt_embeds_mask'],
                num_inference_steps=num_inference_steps,
                guidance_scale=conditions['guidance_scale'],
                return_type='latents',
            )
        
        return latents
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        pass
'''
