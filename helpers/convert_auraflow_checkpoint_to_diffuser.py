import safetensors.torch
import torch

from diffusers import AuraFlowTransformer2DModel, AutoencoderKL, AuraFlowPipeline
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from transformers import T5EncoderModel, T5Tokenizer
from pathlib import Path


def auraflow_checkpoint_to_diffuser(checkpoint_path:str, torch_dtype=torch.float16) -> AuraFlowPipeline:
    checkpoint_path = Path(checkpoint_path)
    repo_id = "fal/AuraFlow"
    (checkpoint_path.parent / 'support-models').mkdir(parents=True, exist_ok=True)

    # not in diffusers yet :(
    transformer = AuraFlowTransformer2DModel.from_single_file(
        checkpoint_path, 
        config=repo_id, 
        subfolder='transformer'
    ).to(torch_dtype)
    
    local_model_pipeline = AuraFlowPipeline.from_pretrained(
        repo_id, 
        cache_dir=checkpoint_path.parent / 'support-models',
        transformer=transformer
    ).to(torch_dtype)

    return local_model_pipeline
