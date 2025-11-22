import asyncio
import gc
import io
import json
import safetensors.torch
import shutil
import socket
import time
import torch
import requests

# import the diffuser pipelines
from diffusers import (AutoencoderKL, AuraFlowPipeline, AuraFlowTransformer2DModel, FluxImg2ImgPipeline, FluxPipeline, FluxTransformer2DModel,
   SD3Transformer2DModel, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, UNet2DConditionModel)
# import the diffuser schedulers
from diffusers import (EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, HeunDiscreteScheduler, PNDMScheduler,
    DDPMScheduler, DDIMScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler, DEISMultistepScheduler)
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.utils import load_image 
from helpers import Async_JSON, auraflow_checkpoint_to_diffuser, vae_pt_to_vae_diffuser, flux_checkpoint_to_diffuser
from huggingface_hub import try_to_load_from_cache
from pathlib import Path
from PIL import Image
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig

def resize_and_crop_centre(images:[Image], new_width:int, new_height:int) -> [Image]:
    """
    Rescales an image to different dimensions without distorting the image.

    images - PIL image or list of PIL images that need to be transformed.
    new_width - int of the desired image width.
    new_height - int of the desired image height.
    """
    if not isinstance(images, list):
        images = [images]

    rescaled_images = []
    for image in images:
        img_width, img_height = image.size
        width_factor, height_factor = new_width/img_width, new_height/img_height
        factor = max(width_factor, height_factor)
        image = image.resize((int(factor*img_width), int(factor*img_height)), Image.LANCZOS)

        img_width, img_height = image.size
        width_factor, height_factor = new_width/img_width, new_height/img_height
        left, top, right, bottom = 0, 0, img_width, img_height
        if width_factor <= 1.5:
            crop_width = int((new_width-img_width)/-2)
            left = left + crop_width
            right = right - crop_width
        if height_factor <= 1.5:
            crop_height = int((new_height-img_height)/-2)
            top = top + crop_height
            bottom = bottom - crop_height
        image = image.crop((left, top, right, bottom))
        rescaled_images.append(image)
    return rescaled_images
    
def download_file(file_path:str or Path, url:str, authorization_token:str=None, retries:int=3) -> Path:
    """
    Download and store a file using chunked download writing with optional authorization. Will retry a certain amount of times if something errors.

    file_path - str or Path to indicate where the file should be downloaded to.
    url - str of where the file will be downloaded from.
    authorization_token - optional str for restricted files. (default None)
    retries - int of how many times to retry the download if it fails. (default 3)
    """
    file_path = Path(file_path)
    for retry in range(retries):
        file_size = file_path.stat().st_size if file_path.is_file() else 0
        headers = {'Range': f'bytes={file_size}-'} if file_size else {}
        if authorization_token:
            headers.update({'Authorization': f'Bearer {authorization_token}'})
        try:
            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                with open(file_path, 'ab') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            return Path(file_path)
        except Exception as e:
            if retry == retries-1:
                raise e

def rmdir(directory):
    """ https://stackoverflow.com/a/49782093 """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


class Model_Manager(object):
    @classmethod
    async def create(cls, civitai_token=None, default_model:str='stable-diffusion-v1-5/stable-diffusion-v1-5', huggingface_token=None, models_path:str='', save_as_4bit:bool=True, save_as_8bit:bool=False, logger=None) -> object:
        """
        Creates and returns an asyncio model_manager object.

        civitai_token - str authenticator token for the civitai api, which is required when downloading some models from the site.
        default_model - str huggingface repo id or a link to a model hosted on civitai (default 'runwayml/stable-diffusion-v1-5')
        huggingface_token - str optional huggingface user access token. (default None)
        models_path - str or path to where the models will be downloaded to. (default '')
        save_as_4bit - bool which sets whether models are reduced to 4bit quantization when they are downloaded or added. (default None)
        save_as_8bit - bool which sets whether models are reduced to 8bit quantization when they are downloaded or added. Is ignored if save_as_4bit is true. (default None)
        logger - logging object (default None)
        """
        self = Model_Manager()
        self.accepted_pipelines = ['AuraFlow', 'Flux', 'SD 1', 'SD 2', 'SD 3', 'SDXL'] 
        self.civitai_token = civitai_token
        self.models_in_queue = []
        self.default_model = default_model
        self.huggingface_token = huggingface_token
        self.models_path = Path(models_path) if models_path else Path('models') 
        self.download_worker = asyncio.create_task(self._download_worker())
        if save_as_4bit and save_as_8bit:
            save_as_8bit = False
        self.save_as_4bit = save_as_4bit
        self.save_as_8bit = save_as_8bit
        self.logger = logger
        self.model_queue = asyncio.Queue(maxsize=0)
        self.models = await self._get_models()
        return self

    async def __call__(self, url_or_repo:str, model_owner:str='system'):
        """
        Attempt to download and add a txt2img model.

        url_or_repo - str of a url pointing to a civitai model, or a main huggingface repo_id.
        model_owner - str identifying the user downloading. (default 'system')
        """
        download_dict = self.get_download_dict(url_or_repo)

        if download_dict['model_name'] in self.models_in_queue:
            raise ValueError(f"{download_dict['model_name']} is already queued for download. Please wait until it finishes.")
        # Owner already exists in the user list.
        if self.get_model_info(download_dict['model_name']) and model_owner in self.get_model_info(download_dict['model_name'])['users']:
            raise ValueError(f"{model_owner} already owns {download_dict['model_name']}.")
        # Add new owner to list of users.
        elif self.get_model_info(download_dict['model_name']) and not model_owner in self.get_model_info(download_dict['model_name'])['users']:
            model_pipeline = self.models.get(download_dict['model_pipeline'])
            model_type = model_pipeline.get(download_dict['model_type'])
            model = model_type.get(download_dict['model_name'])
            model['users'].append(model_owner)
            await Async_JSON.async_save_json(self.models_path / 'diffuser_models.json', self.models)
            return self.get_model_info(download_dict['model_name'])
        
        user_models = await self.get_user_models(model_owner)
        if download_dict['model_name'] in user_models:
                raise ValueError(f"{model_owner} already owns {download_dict['model_name']}.")

        future = asyncio.Future()
        self.models_in_queue.append(download_dict['model_name'])
        await self.model_queue.put((download_dict, future, model_owner))
        return await future

    async def _get_models(self) -> dict:
        """ 
        Initialise the model manager with the last saved list of models and add any local models found in the folder.
        """
        # Make sure the paths exist.
        self.models_path.mkdir(parents=True, exist_ok=True)
        for pipeline in self.accepted_pipelines:
            sub_directories = [self.models_path / pipeline]
            if pipeline in ['Flux', 'SD 1', 'SD 2', 'SDXL']:
                sub_directories.extend([self.models_path / pipeline / 'LORA', self.models_path / pipeline / 'TextualInversion'])
            for path in sub_directories:
                path.mkdir(parents=True, exist_ok=True)
        if not (self.models_path / 'diffuser_models.json').is_file():
            with open(self.models_path / 'diffuser_models.json', 'w') as f:
                json.dump({}, f)
        # Get models downloaded from previous sessions.
        self.models = await Async_JSON.async_load_json(self.models_path / 'diffuser_models.json')

        # Import local diffuser models
        local_diffuser_model_folders = [path.parent for path in self.models_path.rglob('*') if path.name == 'model_index.json']
        for index, folder_path in enumerate(local_diffuser_model_folders):
            if 'snapshots' in str(folder_path):
                folder_path = folder_path.parent.parent
                folder_name = str(folder_path).split('/')[-1].split('--')[-1].replace('-','_').replace('.','_')
            else:
                folder_name = folder_path.stem
            if self.get_model_info(folder_name):
                    continue
            model_pipeline = folder_path.parent.stem
            if model_pipeline not in self.accepted_pipelines:
                continue
            model_dict = {'model_name':folder_name, 'path':local_diffuser_model_folders[index], 'model_pipeline':model_pipeline, 'model_type':'Checkpoint'}
            await self.model_queue.put((model_dict, None, None))
        # Import local checkpoints.
        local_models = [] 
        local_files = [path for path in self.models_path.rglob('*') if path.suffix in ['.ckpt', '.pt', '.safetensors'] and not any(name in path.name for name in ['diffusion_pytorch_model', 'model.fp16.safetensors', 'model.safetensors', '-of-'])]
        for index, file_path in enumerate(local_files):
            # Use directory to determine model info.
            if any(pipeline in str(file_path) for pipeline in self.accepted_pipelines):
                file_name = file_path.name.replace(file_path.suffix, '').replace('.', '_').replace('-', '_')
                Path(local_files[index]).rename(Path(local_files[index].parent, file_name+file_path.suffix))
                if self.get_model_info(file_name) or 'incomplete-download-' in file_name:
                    continue
                # What type of model is it?
                if 'LORA' in file_path.parent.stem:
                    model_type = 'LORA'
                    file_path = file_path.parent
                elif 'TextualInversion' in file_path.parent.stem:
                    model_type = 'TextualInversion'
                    file_path = file_path.parent
                else:
                    model_type = 'Checkpoint'
                # users may have to fix the embed trigger
                embedding_trigger = file_name if model_type == 'TextualInversion' else None

                # What pipeline does it use?
                model_pipeline = file_path.parent.stem
                if model_pipeline not in self.accepted_pipelines:
                    continue

                model_dict = {'model_name':file_name, 'path':local_files[index], 'model_pipeline':model_pipeline, 'model_type':model_type, 'convert_checkpoint_to_diffuser':True, 'embedding_trigger':embedding_trigger}
                await self.model_queue.put((model_dict, None, None))
        # Get the names of all our models to see if the default model is among them.
        model_names = []
        for model_pipeline in self.models:
            model_types = self.models.get(model_pipeline)
            for model in model_types['Checkpoint']:
                model_names.append(model)
        if not self.get_model_info(self.get_download_dict(self.default_model)['model_name']):
            if len(model_names) > 0:
                self.default_model = model_names[0]
            else:
                if self.logger:
                    self.logger.warning(f'No models found at {self.models_path}. Downloading {self.default_model}.')
        return self.models

    async def add_local_model(self, model_name:str, path:str, model_pipeline:str, model_type:str, convert_checkpoint_to_diffuser:bool=False, delete_checkpoint:bool=False, 
        embedding_trigger:str=None, model_owner:str='system', vae_checkpoint_path:str=None) -> dict:
        """
        Uses the provided model details to create and add a model entry to the class, optionally converting it from a checkpoint file to the diffuser format.  
        
        model_name - str identifying the model.
        path - str leading to the model file.
        model_pipeline - str identifying which pipeline the model belongs to.
        model_type - str identifying if the model is a `Checkpoint`, `LORA` or `TextualInversion`.
        convert_checkpoint_to_diffuser - bool switch which sets whether to create a diffuser model from the checkpoint file. (default False)
        delete_checkpoint - bool which sets whether to delete the checkpoint after converting it. (default False)
        embedding_trigger - str which activates the TextualInversion. (default None)
        model_owner - str identifying the user adding the model. (default 'system')
        vae_checkpoint_path - str leading to an external SD vae to convert and add to the pipeline. (default None)
        """
        torch_dtype = torch.float16
        if model_type == 'Checkpoint' and convert_checkpoint_to_diffuser and not 'snapshots' in str(path):
            if model_pipeline == 'AuraFlow':
                raise NotImplementedError('AuraFlowTransformer2DModel.from_single_file is not supported yet.')
                #local_model_pipeline = auraflow_checkpoint_to_diffuser(checkpoint_path=path)
            elif model_pipeline == 'Flux':
                local_model_pipeline = flux_checkpoint_to_diffuser(checkpoint_path=path)
            else:
                pipeline = self.get_pipeline(model_pipeline)
                local_model_pipeline = pipeline.from_single_file(
                    path,
                    torch_dtype=torch_dtype,
                    local_files_only=True
                )
            if delete_checkpoint:
                Path(path).unlink()

            if vae_checkpoint_path:
                vae = vae_pt_to_vae_diffuser(model_pipeline, vae_checkpoint_path)
                (path/'vae').unlink()
                vae.save_pretrained(save_directory=path/'vae')
                vae_checkpoint_path.unlink()

            path = str(self.models_path / model_pipeline / model_name)
            local_model_pipeline.save_pretrained(save_directory=path)
            del local_model_pipeline

        # Quantize the main part of the model to reduce it's size
        is_quantized = Path(path, 'transformer', 'quantized.txt').exists() or Path(path, 'unet', 'quantized.txt').exists()
        if (self.save_as_4bit or self.save_as_8bit) and model_type == 'Checkpoint' and not is_quantized:
            transformer = None
            unet = None

            # These models need their transformer quantized
            if model_pipeline == 'AuraFlow':
                torch_dtype = torch.float16
                quantization_config = DiffusersBitsAndBytesConfig(load_in_4bit=self.save_as_4bit, load_in_8bit=self.save_as_8bit, bnb_4bit_compute_dtype=torch_dtype)
                transformer = AuraFlowTransformer2DModel.from_pretrained(path, subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            if model_pipeline == 'Flux':
                torch_dtype = torch.bfloat16
                quantization_config = DiffusersBitsAndBytesConfig(load_in_4bit=self.save_as_4bit, load_in_8bit=self.save_as_8bit, bnb_4bit_compute_dtype=torch_dtype)
                transformer = FluxTransformer2DModel.from_pretrained(path, subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            if model_pipeline == 'SD3':
                torch_dtype = torch.float16
                quantization_config = DiffusersBitsAndBytesConfig(load_in_4bit=self.save_as_4bit, load_in_8bit=self.save_as_8bit, bnb_4bit_compute_dtype=torch_dtype)
                transformer = SD3Transformer2DModel.from_pretrained(path, subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            if transformer:
                rmdir(Path(path, 'transformer'))
                transformer.save_pretrained(save_directory=Path(path, 'transformer'))
                del transformer
                content = 'savedas4bit' if self.save_as_4bit else 'savedas8bit'
                with open(path/'transformer'/'quantized.txt', 'w') as file: 
                    file.write(content)
            
            # These models need their unet quantized
            if model_pipeline in ['SD1', 'SD2', 'SDXL']:
                torch_dtype = torch.float16
                quantization_config = DiffusersBitsAndBytesConfig(load_in_4bit=self.save_as_4bit, load_in_8bit=self.save_as_8bit, bnb_4bit_compute_dtype=torch_dtype)
                unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', torch_dtype=torch_dtype, quantization_config=quantization_config)
            if unet:
                rmdir((path/'unet'))
                unet.save_pretrained(save_directory=path/'unet')
                del unet
                content = 'savedas4bit' if self.save_as_4bit else 'savedas8bit'
                with open(path/'unet'/'quantized.txt', 'w') as file: 
                    file.write(content)
            torch.cuda.empty_cache()

        # Add model entry information 
        model_pipeline = self.models.setdefault(model_pipeline, {})
        model_type = model_pipeline.setdefault(model_type, {})
        model_entry = model_type.setdefault(model_name, {"path":str(path), "users":[model_owner]})
        if embedding_trigger:
            model_entry.update({'embedding_trigger':embedding_trigger})
        await Async_JSON.async_save_json(self.models_path / 'diffuser_models.json', self.models)

        if self.logger:
                    self.logger.info(f"{model_name} ({path}) added.")
        return model_entry

    def build_pipeline(self, settings_to_pipe):
        """
        Select and build the image generation pipeline.

        settings_to_pipe - dict containing the neccessary settings to build a pipeline and generate an image.  
        """
        seed = settings_to_pipe.pop('seed')
        if seed == -1 or None:
            seed = int(time.time())
        settings_to_pipe['seed'] = seed
        settings_to_pipe = settings_to_pipe.copy()
        model = settings_to_pipe.pop('model')
        model_info = self.get_model_info(model)
        pipe_config = {}
        pipe_config['generator'] = [torch.Generator("cuda").manual_seed(seed+i) for i in range(settings_to_pipe['num_images_per_prompt'])]
       
        def load_lora_and_embeds(lora_and_embeds:list, pipeline_text2image):
            lora_adapters_and_weights = []
            for lora_or_embed in lora_and_embeds:
                index = lora_and_embeds.index(lora_or_embed)
                lora_or_embed_info = self.get_model_info(lora_or_embed.split(':')[0])
                if lora_or_embed_info['model_type'] == 'LORA':
                    lora_adapters_and_weight = lora_or_embed.split(':')
                    lora = lora_adapters_and_weight[0]
                    if len(lora_adapters_and_weight) == 1:
                        # Add a default weight of 1.0 to the lora and correct it in the response...
                        weight = 1.0
                        settings_to_pipe['lora_and_embeds'].remove(lora_or_embed)
                        settings_to_pipe['lora_and_embeds'].insert(index, lora_or_embed+':1.0')
                    else:
                        # ...Or use the provided weight.
                        try:
                            weight = float(lora_adapters_and_weight[-1])
                        except:
                            weight = 1.0
                    pipeline_text2image.load_lora_weights(lora_or_embed_info['path'], adapter_name=lora)
                    lora_adapters_and_weights.append({'lora':lora, 'weight':weight})
                else:
                    pipeline_text2image.load_textual_inversion(pretrained_model_name_or_path=lora_or_embed_info['path'], token=lora_or_embed_info['embedding_trigger'])
                    if not lora_or_embed_info['embedding_trigger'] in settings_to_pipe['prompt']:
                        settings_to_pipe['prompt'] = lora_or_embed_info['embedding_trigger']+' '+settings_to_pipe['prompt']
                    if ':' in lora_or_embed:
                        # Textual Inversion doesn't support weights so correct it for the response
                        settings_to_pipe['lora_and_embeds'].remove(lora_or_embed)
                        settings_to_pipe['lora_and_embeds'].insert(index, lora_or_embed.split(':')[0])
            if lora_adapters_and_weights:
                pipeline_text2image.set_adapters([lora['lora'] for lora in lora_adapters_and_weights], adapter_weights=[lora['weight'] for lora in lora_adapters_and_weights])
            return pipeline_text2image

        def aura_flow(guidance_scale:float, height:int, negative_prompt:str, num_inference_steps:int, num_images_per_prompt:int, prompt:str, width:int, model_info:dict=model_info, **kwargs):
            """
            Build a pipeline for an AuraFlow model.
            """
            pipeline = self.get_pipeline(model_info['model_pipeline'])
            torch_dtype = torch.float16

            # get our encoded prompt latents so we don't have to load the text encoders during generation
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch.bfloat16,
                variant='fp16',
                transformer=None,
                vae=None
            ).to('cuda')

            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipeline_text2image.encode_prompt(prompt=prompt, negative_prompt=negative_prompt, 
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=512, device='cuda')
            del pipeline_text2image

            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                torch_dtype=torch.bfloat16,
                variant='fp16'
            ).to('cuda')

            pipe_config['guidance_scale'] = guidance_scale
            pipe_config['height'] = height
            pipe_config['num_inference_steps'] = num_inference_steps
            pipe_config['negative_prompt_attention_mask'] = negative_prompt_attention_mask
            pipe_config['negative_prompt_embeds'] = negative_prompt_embeds
            pipe_config['prompt_attention_mask'] = prompt_attention_mask
            pipe_config['prompt_embeds'] = prompt_embeds
            pipe_config['width'] = width

            return pipeline_text2image, pipe_config
           
        def flux(guidance_scale:float, height:int, num_inference_steps:int, num_images_per_prompt:int, prompt:str, width:int, init_image:str=None, init_strength:float=None, lora_and_embeds:list=None, 
            model_info:dict=model_info, negative_prompt:str=None, hires_fix:bool=None, hires_strength:float=None, **kwargs):
            """
            Build a pipeline for a Flux model.
            """
            torch_dtype = torch.bfloat16

            model_pipeline = model_info['model_pipeline'] 
            if init_image:
                model_pipeline += "Img2Img"
            pipeline = self.get_pipeline(model_pipeline)
            
            # get our encoded prompt latents so we don't have to load the text encoders during generation
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                transformer=None,
                vae=None
            ).to('cuda')

            prompt_embeds, pooled_prompt_embeds, _ = pipeline_text2image.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=512, device='cuda')
            negative_prompt_embeds, negative_pooled_prompt_embeds, _ = pipeline_text2image.encode_prompt(prompt=negative_prompt, prompt_2=None, max_sequence_length=512, device='cuda')
            del pipeline_text2image

            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=torch_dtype,
            ).to('cuda')

            # Memory and speed optmisation
            pipeline_text2image.vae.enable_slicing()
            pipeline_text2image.vae.enable_tiling()

            # Load additional networks
            if lora_and_embeds:
                pipeline_text2image = load_lora_and_embeds(lora_and_embeds, pipeline_text2image)
            
            # Finalise pipe_config
            if hires_fix:
                # Prepare pipeline for hires_fix by shrinking initial dimensions.
                model_res = 768
                while width > model_res and height > model_res:
                    width -= 8
                    height -= 8
            if init_image:
                # prepare init image
                init_image = load_image(init_image)
                init_image = resize_and_crop_centre(init_image, width, height)
                pipe_config['image'] = init_image * num_images_per_prompt
                pipe_config['strength'] = init_strength
            pipe_config['guidance_scale'] = guidance_scale
            pipe_config['height'] = height
            pipe_config['num_inference_steps'] = num_inference_steps
            pipe_config['num_images_per_prompt'] = num_images_per_prompt
            pipe_config['negative_prompt_embeds'] = negative_prompt_embeds
            pipe_config['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
            pipe_config['prompt_embeds'] = prompt_embeds
            pipe_config['pooled_prompt_embeds'] = pooled_prompt_embeds
            pipe_config['width'] = width

            return pipeline_text2image, pipe_config

        def stable_diffusion(clip_skip:int, guidance_scale:float, height:int, num_inference_steps:int, num_images_per_prompt:int, prompt:str, scheduler:str, width:int, init_image:str=None, init_strength:float=None, 
            lora_and_embeds:list=None, model_info:dict=model_info, negative_prompt:str=None, hires_fix:bool=None, hires_strength:float=None, **kwargs):
            """
            Build a pipeline for a pre SD 3 stable diffusion model.
            """
            torch_dtype = torch.float16
            model_pipeline = model_info['model_pipeline'] 
            if init_image:
                model_pipeline += "Img2Img"
            pipeline = self.get_pipeline(model_pipeline)
            

            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                safety_checker=None,
                variant='fp16'
            ).to('cuda')

            # Memory and speed optimisation
            pipeline_text2image.enable_attention_slicing()
            pipeline_text2image.vae.enable_slicing()

            # Use the specified scheduler
            pipeline_scheduler = self.get_scheduler(model_info['model_pipeline'], scheduler)
            pipeline_text2image.scheduler = pipeline_scheduler.from_config(pipeline_text2image.scheduler.config)
            # implement non-k schedulers
            pipeline_text2image.scheduler.use_karras_sigmas=True
            
            # Load additional networks
            if lora_and_embeds:
                pipeline_text2image = load_lora_and_embeds(lora_and_embeds, pipeline_text2image)

            # Finalise pipe_config
            if hires_fix:
                # Prepare pipeline for hires_fix by shrinking initial dimensions.
                model_res = pipeline_text2image.unet.config.sample_size * pipeline_text2image.vae_scale_factor
                while width > model_res and height > model_res:
                    width -= 8
                    height -= 8
            if init_image:
                # prepare init image
                init_image = load_image(init_image)
                init_image = resize_and_crop_centre(init_image, width, height)
                pipe_config['image'] = init_image * num_images_per_prompt
                pipe_config['strength'] = init_strength
            pipe_config['clip_skip'] = clip_skip
            pipe_config['guidance_scale'] = guidance_scale
            pipe_config['height'] = height
            pipe_config['num_inference_steps'] = num_inference_steps
            pipe_config['num_images_per_prompt'] = num_images_per_prompt
            pipe_config['negative_prompt'] = negative_prompt
            pipe_config['prompt'] = prompt
            pipe_config['width'] = width

            return pipeline_text2image, pipe_config

        def stable_diffusion_3(clip_skip:int, guidance_scale:float, height:int, num_inference_steps:int, num_images_per_prompt:int, prompt:str, width:int, negative_prompt:str=None, model_info:dict=model_info,
            **kwargs):
            """
            Build a pipeline for a stable diffusion 3 model.
            """
            torch_dtype = torch.float16
            pipeline = self.get_pipeline(model_info['model_pipeline'])

            # encode the prompt
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                transformer=None,
                vae=None,
                variant='fp16'
            ).to('cuda')
            (
                prompt_embeds, 
                negative_prompt_embeds, 
                pooled_prompt_embeds, 
                negative_pooled_prompt_embeds
            ) = pipeline_text2image.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt, negative_prompt=negative_prompt, 
                negative_prompt_2=negative_prompt, negative_prompt_3=negative_prompt, num_images_per_prompt= num_images_per_prompt, max_sequence_length=512, 
                clip_skip=clip_skip, device='cuda')
            del pipeline_text2image
            
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                text_encoder_2=None,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch_dtype,
                variant='fp16'
            ).to('cuda')

            # Finalise pipe_config
            pipe_config['guidance_scale'] = guidance_scale
            pipe_config['height'] = height
            pipe_config['num_inference_steps'] = num_inference_steps
            pipe_config['num_images_per_prompt'] = num_images_per_prompt
            pipe_config['negative_prompt_embeds'] = negative_prompt_embeds
            pipe_config['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
            pipe_config['prompt_embeds'] = prompt_embeds
            pipe_config['pooled_prompt_embeds'] = pooled_prompt_embeds
            pipe_config['width'] = width

            return pipeline_text2image, pipe_config

        pipeline_text2image = {
            'AuraFlow': aura_flow,
            'Flux': flux,
            'SD 1': stable_diffusion,
            'SD 2': stable_diffusion,
            'SD 3': stable_diffusion_3,
            'SDXL': stable_diffusion
        }.get(model_info['model_pipeline'])
        return pipeline_text2image

    def get_download_dict(self, url_or_repo:str):
        """
        Get the necessary info to download and add a model.

        url_or_repo - str leading to the civitai model, or a huggingface model repo.
        """
        def internet_connection():
            try:
                socket.create_connection(('1.1.1.1', 53))
                return True
            except OSError:
                pass
            return False
        if not internet_connection():
            raise OSError(f'Cannot download {url_or_repo} due to not having an internet connection.')

        download_dict = {}
        # If it's not a civitai address, assume it's one of the allowed huggingface repos.
        if not 'civitai.com/models' in url_or_repo: 
            model_name = url_or_repo.split('/')[-1].replace('.', '_').replace('-','_')
            if 'AuraFlow' in url_or_repo:
                model_pipeline = 'AuraFlow'
            elif 'FLUX' in url_or_repo:
                model_pipeline = 'Flux'
            elif 'stable-diffusion-v1-5' in url_or_repo:
                model_pipeline = 'SD 1'
            elif 'stable-diffusion-2' in url_or_repo:
                model_pipeline = 'SD 2'
            elif 'stable-diffusion-3' in url_or_repo:
                model_pipeline = 'SD 3'
            elif 'stable-diffusion-xl' in url_or_repo:
                model_pipeline = 'SDXL'
            else:
                raise ValueError(f'Huggingface link leads to an unsupported model. Accepted models are {self.accepted_pipelines}.')
            download_dict.update({'model_pipeline':model_pipeline, 'model_type':"Checkpoint", 'model_name':model_name, 'url_or_repo':url_or_repo})
        else:
            sub_model_id = url_or_repo.split('/')[-1]
            model_id = url_or_repo.split('/')[-2]
            if model_id == 'models':
                model_id = sub_model_id.split('?')[0]
                sub_model_id = sub_model_id.replace(model_id, '')
            api_url = f'https://civitai.com/api/v1/models/{model_id}'
            response = requests.get(api_url)
            response = response.json()
            # Which version of the model do we want?
            if '?modelVersionId=' in sub_model_id:
                try:
                    model_version = sub_model_id[sub_model_id.rfind('?modelVersionId=')+16:]
                    chosen_model = [version for version in response['modelVersions'] if str(version['id']) == model_version]
                except:
                    raise ValueError('Unable to retreive model version from the url.')
                if len(chosen_model) == 1:
                    chosen_model = chosen_model[0]
                else:
                    chosen_model = response['modelVersions'][0]
            else:
                chosen_model = response['modelVersions'][0]
            
            if response['type'] in ['Checkpoint', 'TextualInversion', 'LORA']:
                chosen_file = [file for file in chosen_model['files'] if file['type'] == 'Model'][0]
                vae_list = [file for file in chosen_model['files'] if file['type'] == 'VAE']
                file_name = chosen_file['name'][:chosen_file['name'].rfind('.')].replace('.','_')+chosen_file['name'][chosen_file['name'].rfind('.'):]
                url_or_repo = chosen_model['downloadUrl']
                
                # Get the model's pipeline.    
                if 'AuraFlow' in chosen_model['baseModel']:
                    model_pipeline = 'AuraFlow'
                elif 'Flux' in chosen_model['baseModel']:
                    model_pipeline = 'Flux'
                elif 'SD 1' in chosen_model['baseModel']:
                    model_pipeline = 'SD 1'
                elif 'SD 2' in chosen_model['baseModel']:
                    model_pipeline = 'SD 2'
                elif 'SD 3' in chosen_model['baseModel']:
                    model_pipeline = 'SD 3'
                elif 'SDXL' in chosen_model['baseModel']:
                    model_pipeline = 'SDXL'
                else:
                    raise ValueError(f'Civitai link leads to an unsupported model. Accepted models are {self.accepted_pipelines}.')

                # include the vae if a model is listed with it
                if vae_list:
                    latest_vae = vae_list[0]
                    vae_name = latest_vae['name']
                    vae_download_path = self.models_path / model_pipeline / vae_name
                    vae_url = latest_vae['downloadUrl']
                else:
                    vae_download_path = None
                    vae_url = None

                embedding_trigger = chosen_model['trainedWords'][0] if response['type'] == 'TextualInversion' else None

                download_dict.update({'model_name':file_name,'model_pipeline':model_pipeline, 'model_type':response['type'], 'url_or_repo':url_or_repo,
                    'vae_download_path':vae_download_path, 'vae_url':vae_url, 'embedding_trigger':embedding_trigger})
        return download_dict

    def get_model_info(self, model_name:str) -> dict:
        """
        Search existing models to see if exists, and return it's settings_to_pipe.

        model_name - str identifying the model. 
        """
        model_info = None
        for model_pipeline in self.models:
            model_types = self.models.get(model_pipeline)
            for model_type in model_types:
                models_of_type = self.models.get(model_pipeline).get(model_type)
                for stored_model in models_of_type:
                    if model_name == stored_model:
                        model_dict = models_of_type.get(stored_model)
                        model_info = {'model_name': model_name, 'model_pipeline': model_pipeline, 'model_type':model_type, 'path': model_dict['path'], 'users': model_dict['users']}
                        if 'TextualInversion' in model_type:
                            model_info.update({'embedding_trigger': model_dict['embedding_trigger']})
        return model_info

    def get_pipeline(self, pipeline:str):
        """ 
        Retrieve the specific pipeline for the model.

        pipeline - str for the corresponding pipeline object.
        """
        pipelines = {
            "AuraFlow": AuraFlowPipeline,
            "Flux": FluxPipeline,
            "FluxImg2Img": FluxImg2ImgPipeline,
            "SD 1": StableDiffusionPipeline,
            "SD 1Img2Img": StableDiffusionImg2ImgPipeline,
            "SD 2": StableDiffusionPipeline,
            "SD 2Img2Img": StableDiffusionImg2ImgPipeline,
            "SD 3": StableDiffusion3Pipeline,
            "SDXL": StableDiffusionXLPipeline,
            "SDXLImg2Img": StableDiffusionXLImg2ImgPipeline
        }
        return pipelines.get(pipeline)

    def get_scheduler(self, pipeline:str, scheduler:str):
        """
        Retrieve the specified scheduler.
        """
        # there isnt much for flowmatch atm
        schedulers = {
            ('SD 1', 'SD 2', 'SDXL'): {
                "euler_ancestral": EulerAncestralDiscreteScheduler,
                "dpm_solver_multistep": DPMSolverMultistepScheduler,
                "dpm_solver_singlestep": DPMSolverSinglestepScheduler,
                "heun": HeunDiscreteScheduler,
                "pndm": PNDMScheduler,
                "ddpm": DDPMScheduler,
                "ddim": DDIMScheduler,
                "k_dpm_2": KDPM2DiscreteScheduler,
                "k_dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
                "dpm_solver_sde": DPMSolverSDEScheduler,
                "unipc_multistep": UniPCMultistepScheduler,
                "deis_multistep": DEISMultistepScheduler
                }
            }
        for model_pipeline in schedulers:
            if pipeline in model_pipeline:
                return schedulers.get(model_pipeline).get(scheduler)
        raise ValueError(f'Scheduler, {scheduler} for {pipeline} not found.')

    async def get_user_models(self, model_owner:str='system') -> list:
        """ 
        Retrieve any models under the owner.

        model_owner - str of which user the models are being searched for.
        """
        user_models = []
        for model_pipeline in self.models:
            model_types = self.models.get(model_pipeline)
            for model_type in model_types:
                models_of_type = model_types.get(model_type)
                for model in models_of_type:
                    model_dict = models_of_type.get(model)
                    if not model_owner in model_dict['users']:
                        pass
                    else:
                        user_models.append(model)
        return user_models

    def list_models(self) -> dict:
        models = {}
        # go through everything in alphabetical order
        model_pipelines = [pipeline for pipeline in self.models]
        model_pipelines.sort()
        
        for model_pipeline in model_pipelines:
            models.update({model_pipeline:{}})
            model_types = [model_type for model_type in self.models.get(model_pipeline)]
            model_types.sort()

            # go through model types
            for model_type in model_types:
                models_of_type = [model for model in self.models.get(model_pipeline).get(model_type)]
                models_of_type = sorted(models_of_type, key=str.casefold)
                models[model_pipeline].update({model_type:models_of_type})

        return models

    async def remove_model(self, model_name:str, model_owner:str='system') -> None:
        """
        Remove user ownership over the model, deleting it if unused.
        """
        model_info = self.get_model_info(model_name)
        model_pipeline = self.models.get(model_info['model_pipeline'])
        model_type = model_pipeline.get(model_info['model_type'])
        model_entry = model_type.get(model)
        
        model_entry['users'].remove(model_owner)

        # Remove the model entry and delete it if no one is using it.
        if len(model_entry['users']) == 0:
            model_type.pop(model)
            if '/snapshots/' in model_info['path']:
                model_path = Path(model_info['path']).parent.parent
            else: 
                model_path = Path(model_info['path'])

            if 'Checkpoint' in model_info['model_type']: 
                shutil.rmtree(model_path)
            else:
                model_path.unlink()

            # Remove the model type if there are no models.
            if len(model_type) <= 0:
                model_pipeline.pop(model_info['model_type'])
                # Remove the pipeline if there are no base models.
                if not model_pipeline.get('Checkpoint', None):
                    self.models.pop(model_info['model_pipeline'])

        await Async_JSON.async_save_json(self.models_path / 'diffuser_models.json', self.models)

    async def _download_worker(self) -> None:
        """
        Asyncio download worker that retrieves items from the download queue.
        """
        def download_model(model_name:str, model_pipeline:str, model_type:str, url_or_repo:str, vae_download_path:str=None, vae_url:str=None, 
            embedding_trigger:str=None) -> dict:
            """
            Uses a diffusers pipeline to manage the download if its from huggingface, else download it from civitai and turn that checkpoint file into the diffusers format.
            """
            if not 'civitai.com' in url_or_repo:
                # Use the diffuser pipeline to manage the model download of a huggingface diffuser model.
                pipeline = self.get_pipeline(model_pipeline)

                # Pipeline sourced from the fp16 variant
                if model_pipeline not in ['Flux']:
                    local_model_pipeline = pipeline.from_pretrained(
                        url_or_repo,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                        cache_dir=self.models_path / model_pipeline,
                        token=self.huggingface_token
                    )
                else:
                    local_model_pipeline = pipeline.from_pretrained(
                        url_or_repo,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        cache_dir=self.models_path / model_pipeline,
                        token=self.huggingface_token
                    )

                model_index_path = try_to_load_from_cache(url_or_repo, filename='model_index.json', cache_dir=self.models_path / model_pipeline)
                diffuser_path = Path(model_index_path).parent
                del local_model_pipeline
                torch.cuda.empty_cache()
                model_dict = {'model_name':model_name, 'model_pipeline':model_pipeline, 'model_type':model_type, 'path':str(diffuser_path)}
            else:
                # Download the model from civitai and turn it into a diffuser model.
                temp_model_name = 'incomplete-download-'+model_name
                download_path = self.models_path / model_pipeline / (temp_model_name if model_type == 'Checkpoint' else Path(model_type) / temp_model_name)
                model_path = download_file(download_path, url_or_repo, authorization_token=self.civitai_token)
                model_path = download_path
                model_path = model_path.rename(model_path.parent / model_name)
                model_dict = {'model_pipeline':model_pipeline, 'model_type':model_type, 'model_name': model_name.replace('.safetensors', '').replace('.ckpt','').replace('.pt', ''), 
                    'path':str(model_path)}     
                if 'Checkpoint' in model_type and vae_url:
                    vae_checkpoint_path = download_file(vae_download_path, vae_url, authorization_token=self.civitai_token)
                    model_dict.update({'vae_checkpoint_path': vae_checkpoint_path})    
                if 'TextualInversion' in model_type:
                    model_dict.update({'embedding_trigger': embedding_trigger})
                    
            self.models_in_queue.remove(model_name)
            return model_dict
            
        async def process_download(download_dict:dict, future, model_owner:str):
            """
            Download all the files.
            """
            if self.logger:
                    self.logger.info(f"Downloading {download_dict['model_name']} from {download_dict['url_or_repo']}...")
            model_dict = await asyncio.to_thread(download_model, **download_dict)
            # Save the entry to the class dict
            await self.add_local_model(**model_dict, convert_checkpoint_to_diffuser=True, delete_checkpoint=True, model_owner=model_owner)
            future.set_result(model_dict)
            
        while True:
            try:
                model_or_download_dict, future, model_owner = await self.model_queue.get()
                if model_or_download_dict.get('url_or_repo', None):
                    await process_download(model_or_download_dict, future, model_owner)
                else: 
                    await self.add_local_model(**model_or_download_dict)    
            except Exception as exception:
                if self.logger:
                    self.logger.error(f"Model manager encountered an error while adding {model_or_download_dict['model_name']}:\n{exception}")
                if future:
                    future.set_exception(exception)
