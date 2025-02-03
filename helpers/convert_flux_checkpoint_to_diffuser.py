import safetensors.torch
import torch

from diffusers import AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from pathlib import Path


def flux_checkpoint_to_diffuser(checkpoint_path:str, torch_dtype=torch.float16) -> FluxPipeline:
    checkpoint_path = Path(checkpoint_path)
    original_state_dict = safetensors.torch.load_file(checkpoint_path)
    has_guidance = any("guidance" in k for k in original_state_dict)
    repo_id = "black-forest-labs/FLUX.1-schnell" if not has_guidance else "black-forest-labs/FLUX.1-dev"
    (checkpoint_path.parent / 'support-models').mkdir(parents=True, exist_ok=True)

    # find a way to reduce the transformer memory footprint (~50gb!)
    transformer = FluxTransformer2DModel(guidance_embeds=True).from_single_file(
        checkpoint_path, 
        config=repo_id, 
        subfolder='transformer'
    ).to(torch_dtype)

    config = AutoencoderKL.load_config(repo_id, subfolder="vae")
    vae = AutoencoderKL.from_config(config).to(torch_dtype)
    converted_vae_state_dict = convert_ldm_vae_checkpoint(original_state_dict, vae.config)
    vae.load_state_dict(converted_vae_state_dict, strict=True)

    local_model_pipeline = FluxPipeline.from_pretrained(
        repo_id, 
        cache_dir=checkpoint_path.parent / 'support-models',
        transformer=transformer, 
        vae=vae
    ).to(torch_dtype)

    return local_model_pipeline
