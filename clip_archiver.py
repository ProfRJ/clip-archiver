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
from diffusers import (AutoencoderKL, AuraFlowPipeline, AuraFlowTransformer2DModel, BitsAndBytesConfig, FluxImg2ImgPipeline, FluxPipeline, FluxTransformer2DModel, 
   SD3Transformer2DModel, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, UNet2DConditionModel)
# import the diffuser schedulers
from diffusers import (EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, HeunDiscreteScheduler, PNDMScheduler,
    DDPMScheduler, DDIMScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler, DEISMultistepScheduler)
from diffusers.utils import load_image 
from helpers import Async_JSON, auraflow_checkpoint_to_diffuser, vae_pt_to_vae_diffuser, flux_checkpoint_to_diffuser
from huggingface_hub import try_to_load_from_cache
from pathlib import Path
from PIL import Image
from transformers import T5EncoderModel

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

class Model_Manager(object):
    @classmethod
    async def create(cls, civitai_token='', default_model:str='stable-diffusion-v1-5/stable-diffusion-v1-5', models_path:str='', logger=None) -> None:
        """
        Creates and returns an asyncio model_manager object.

        civitai_token - An authenticator token for the civitai api, which is required when downloading some models from the site.
        models_path - str or pathlike object with the intended location for the models. (default '')
        default_model - huggingface repo id or a link to a model hosted on civitai (default 'runwayml/stable-diffusion-v1-5')
        logger - logging object (default None)
        """
        self = Model_Manager()
        self.accepted_pipelines = ['AuraFlow', 'Flux', 'SD 1', 'SD 2', 'SD 3', 'SDXL'] 
        self.civitai_token = civitai_token
        self.models_in_queue = []
        self.default_model = default_model
        self.models_path = Path(models_path) if models_path else Path('models') 
        self.download_worker = asyncio.create_task(self._download_worker())
        self.model_queue = asyncio.Queue(maxsize=0)
        self.logger = logger
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
            model = model_pipeline.get(download_dict['model_name'])
            model['users'].append(context.author.id)
            await Async_JSON.async_save_json(self.models_path / 'diffuser_models.json', self.clip_archiver.models)
            return model
        
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
        local_files = [path for path in self.models_path.rglob('*') if path.suffix in ['.ckpt', '.pt', '.safetensors'] and not any(name in path.name for name in ['diffusion_pytorch_model', 'model.fp16.safetensors', 'model.safetensors'])]
        for index, file_path in enumerate(local_files):
            # Use directory to determine model info.
            if any(pipeline in str(file_path) for pipeline in self.accepted_pipelines):
                file_name = file_path.name.replace(file_path.suffix, '')
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
                    self.logger.warning(f'No models found at {self.models_path}.')
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

        if model_type == 'Checkpoint' and convert_checkpoint_to_diffuser and Path(path).is_file():
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
            path = str(self.models_path / model_pipeline / model_name)
            local_model_pipeline.save_pretrained(save_directory=path)
            del local_model_pipeline
            torch.cuda.empty_cache()

            if vae_checkpoint_path:
                vae_subfolder = path+'/vae'
                vae_pt_to_vae_diffuser(model_pipeline, vae_checkpoint_path, vae_subfolder)
                if delete_checkpoint:
                    vae_checkpoint_path.unlink()

        model_pipeline = self.models.setdefault(model_pipeline, {})
        model_type = model_pipeline.setdefault(model_type, {})
        model_entry = model_type.setdefault(model_name, {"path":str(path), "users":[model_owner]})
        if embedding_trigger:
            model_entry.update({'embedding_trigger':embedding_trigger})
        await Async_JSON.async_save_json(self.models_path / 'diffuser_models.json', self.models)

        if self.logger:
                    self.logger.info(f"{model_name} ({path}) added.")
        return model_entry

    def build_pipeline(self, settings_to_pipe:dict):
        """
        Select and build the image generation pipeline.

        settings_to_pipe - dict containing the neccessary settings to build a pipeline and generate an image.  
        """
        seed = settings_to_pipe.pop('seed')
        if not seed:
            seed = int(time.time())
        pipe_config = settings_to_pipe.copy()
        settings_to_pipe['seed'] = seed

        pipe_config['generator'] = [torch.Generator("cuda").manual_seed(seed+i) for i in range(pipe_config['batch_size'])]
        model = pipe_config.pop('model')
        model_info = self.get_model_info(model)
       
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

        def aura_flow(model_info:dict=model_info, pipe_config:dict=pipe_config):
            pipeline = self.get_pipeline(model_info['model_pipeline'])
            torch_dtype = torch.float16
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch.float16,
                variant='fp16',
                transformer=None,
                vae=None
            ).to('cuda')
            (
                pipe_config['prompt_embeds'], 
                pipe_config['prompt_attention_mask'], 
                pipe_config['negative_prompt_embeds'], 
                pipe_config['negative_prompt_attention_mask']
            ) = pipeline_text2image.encode_prompt(prompt=pipe_config.pop('prompt'), negative_prompt=pipe_config.pop('negative_prompt', None), num_images_per_prompt=pipe_config.pop('batch_size'), max_sequence_length=512, 
                device='cuda')
            del pipeline_text2image

            # quantize the transformer to 4bit precision to reduce its memory footprint
            transformer = AuraFlowTransformer2DModel.from_pretrained(model_info['path'], subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                torch_dtype=torch.float16,
                transformer=None,
                variant='fp16'
            ).to('cuda')
            pipeline_text2image.transformer = transformer
            del transformer

            return pipeline_text2image, pipe_config

        def stable_diffusion(model_info:dict=model_info, pipe_config:dict=pipe_config, hires_run=False):
            """
            Build a pipeline for a pre SD 3 stable diffusion model.
            """
            torch_dtype = torch.float16
            init_image = pipe_config.pop('init_image', None)
            hires_run = pipe_config.pop('hires_run', False)
            pipeline = self.get_pipeline(model_info['model_pipeline'] if not init_image and not hires_run else model_info['model_pipeline']+'Img2Img')
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
            
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                safety_checker=None,
            ).to('cuda')
            pipeline_text2image.unet = UNet2DConditionModel.from_pretrained(model_info['path'], subfolder='unet', torch_dtype=torch_dtype, quantization_config=quantization_config)
            
            # Memory and speed optimisation
            pipeline_text2image.enable_attention_slicing()
            pipeline_text2image.enable_vae_slicing()

            # Use the specified scheduler
            pipeline_scheduler = self.get_scheduler(model_info['model_pipeline'], pipe_config.pop('scheduler', None))
            pipeline_text2image.scheduler = pipeline_scheduler.from_config(pipeline_text2image.scheduler.config)
            pipeline_text2image.scheduler.use_karras_sigmas=True
            
            # Load additional networks
            lora_and_embeds = pipe_config.pop('lora_and_embeds', None)
            if lora_and_embeds:
                pipeline_text2image = load_lora_and_embeds(lora_and_embeds, pipeline_text2image)

            # Finalise pipe_config
            pipe_config['num_images_per_prompt'] = pipe_config.pop('batch_size')
            if not hires_run:
                if pipe_config.pop('hires_fix', False):
                    # Prepare pipeline for hires_fix by shrinking initial dimensions.
                    model_res = pipeline_text2image.unet.config.sample_size * pipeline_text2image.vae_scale_factor
                    while pipe_config['width'] > model_res and pipe_config['height'] > model_res:
                        pipe_config['width'] -= 8
                        pipe_config['height'] -= 8
                    pipe_config.pop('hires_strength')
                if init_image:
                    # prepare init image
                    init_image = load_image(init_image)
                    init_image = resize_and_crop_centre(init_image, pipe_config['width'], pipe_config['height'])
                    pipe_config['image'] = init_image * pipe_config['num_images_per_prompt']
                    pipe_config['strength'] = pipe_config.pop('init_strength') 
            else:
                pipe_config.pop('init_strength', None) 
                pipe_config.pop('hires_fix')   
                pipe_config['image'] = init_image
                pipe_config['strength'] = pipe_config.pop('hires_strength')

            return pipeline_text2image, pipe_config

        def stable_diffusion_3(model_info:dict=model_info, pipe_config:dict=pipe_config):
            """
            Build a pipeline for a stable diffusion 3 model.
            """
            torch_dtype = torch.float16
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
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
                pipe_config['prompt_embeds'], 
                pipe_config['negative_prompt_embeds'], 
                pipe_config['pooled_prompt_embeds'], 
                pipe_config['negative_pooled_prompt_embeds']
            ) = pipeline_text2image.encode_prompt(prompt=pipe_config.get('prompt'), prompt_2=pipe_config.get('prompt'), prompt_3=pipe_config.pop('prompt'), negative_prompt=pipe_config.get('negative_prompt', None), 
                negative_prompt_2=pipe_config.get('negative_prompt', None), negative_prompt_3=pipe_config.pop('negative_prompt', None), num_images_per_prompt= pipe_config.pop('batch_size'), max_sequence_length=512, 
                clip_skip=pipe_config.pop('clip_skip', None), device='cuda')
            del pipeline_text2image

            # quantize the transformer to 4bit precision to reduce its memory footprint
            transformer = SD3Transformer2DModel.from_pretrained(model_info['path'], subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                text_encoder_2=None,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch_dtype,
                transformer=transformer,
                variant='fp16'
            ).to('cuda')
            del transformer

            return pipeline_text2image, pipe_config

        def flux(model_info:dict=model_info, pipe_config:dict=pipe_config):
            """
            Build a pipeline for an Flux model.
            """
            torch_dtype = torch.bfloat16
            init_image = pipe_config.pop('init_image', None)
            hires_run = pipe_config.pop('hires_run', False)
            # FluxImg2ImgPipeline is lacking in attributes so we need to make the distinction a lot of times.
            pipeline = self.get_pipeline(model_info['model_pipeline'] if not init_image and not hires_run else model_info['model_pipeline']+'Img2Img')
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

            
            # get our encoded prompt latents so we don't have to load the text encoders during generation
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                transformer=None,
                vae=None
            ).to('cuda')
            pipe_config['prompt_embeds'], pipe_config['pooled_prompt_embeds'], text_ids = pipeline_text2image.encode_prompt(prompt=pipe_config.pop('prompt'), prompt_2=None, max_sequence_length=512, device='cuda')
            if not init_image and not hires_run:
                pipe_config['negative_prompt_embeds'], pipe_config['negative_pooled_prompt_embeds'], text_ids = pipeline_text2image.encode_prompt(prompt=pipe_config.pop('negative_prompt'), prompt_2=None, max_sequence_length=512, device='cuda')
            # img2img pipeline doesnt accept negative prompt
            else:
                pipe_config.pop('negative_prompt')
            del pipeline_text2image

            # quantize the transformer to 4bit precision to reduce its memory footprint
            transformer = FluxTransformer2DModel.from_pretrained(model_info['path'], subfolder='transformer', torch_dtype=torch_dtype, quantization_config=quantization_config)
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=torch_dtype,
                transformer=transformer
            ).to('cuda')
            del transformer

            # Memory and speed optmisation
            # doesnt work with the img2img pipeline?
            if not init_image and not hires_run:
                pipeline_text2image.enable_vae_slicing()
                pipeline_text2image.enable_vae_tiling()

            # Load additional networks
            lora_and_embeds = pipe_config.pop('lora_and_embeds', None)
            if lora_and_embeds:
                pipeline_text2image = load_lora_and_embeds(lora_and_embeds, pipeline_text2image)
            
            # Finalise pipe_config
            pipe_config['num_images_per_prompt'] = pipe_config.pop('batch_size')
            if not hires_run:
                if pipe_config.pop('hires_fix', False):
                    # Prepare pipeline for hires_fix by shrinking initial dimensions.
                    model_res = 1024
                    while pipe_config['width'] > model_res and pipe_config['height'] > model_res:
                        pipe_config['width'] -= 8
                        pipe_config['height'] -= 8
                    pipe_config.pop('hires_strength')
                if init_image:
                    # prepare init image
                    init_image = load_image(init_image)
                    init_image = resize_and_crop_centre(init_image, pipe_config['width'], pipe_config['height'])
                    pipe_config['image'] = init_image * pipe_config['num_images_per_prompt']
                    pipe_config['strength'] = pipe_config.pop('init_strength') 
            else:
                pipe_config.pop('init_strength', None) 
                pipe_config.pop('hires_fix')   
                pipe_config['image'] = init_image
                pipe_config['strength'] = pipe_config.pop('hires_strength')
            return pipeline_text2image, pipe_config

        pipeline_text2image = {
            'AuraFlow': aura_flow,
            'SD 1': stable_diffusion,
            'SD 2': stable_diffusion,
            'SD 3': stable_diffusion_3,
            'SDXL': stable_diffusion,
            'Flux': flux
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
            elif 'stable-diffusion-v1-5' in url_or_repo:
                model_pipeline = 'SD 1'
            elif 'stable-diffusion-2' in url_or_repo:
                model_pipeline = 'SD 2'
            elif 'stable-diffusion-3' in url_or_repo:
                model_pipeline = 'SD 3'
            elif 'stable-diffusion-xl' in url_or_repo:
                model_pipeline = 'SDXL'
            elif 'FLUX' in url_or_repo:
                model_pipeline = 'Flux'
            else:
                raise ValueError(f'Huggingface link leads to an unsupported model. Accepted models are {self.accepted_pipelines}.')
            download_dict.update({'model_pipeline':model_pipeline, 'model_type':"Checkpoint", 'model_name':model_name, 'url_or_repo':url_or_repo})
        else:
            model_id = url_or_repo.split('/')[-1]
            api_url = f'https://civitai.com/api/v1/models/{model_id}'
            response = requests.get(api_url)
            response = response.json()
            # Which version of the model do we want?
            if '?modelVersionId=' in model_id:
                try:
                    model_version = model_id[model_id.rfind('?modelVersionId=')+16:]
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
                    raise ValueError('AuraFlowTransformer2DModel.from_single_file is not supported yet.')
                    model_pipeline = 'AuraFlow'
                elif 'SD 1' in chosen_model['baseModel']:
                    model_pipeline = 'SD 1'
                elif 'SD 2' in chosen_model['baseModel']:
                    model_pipeline = 'SD 2'
                elif 'SD 3' in chosen_model['baseModel']:
                    # not a chance :P
                    model_pipeline = 'SD 3'
                elif 'SDXL' in chosen_model['baseModel']:
                    model_pipeline = 'SDXL'
                elif 'Flux' in chosen_model['baseModel']:
                    model_pipeline = 'Flux'
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
        Search existing models to see if exists, and return it's settings.

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
                        cache_dir=self.models_path / model_pipeline
                    )
                else:
                    local_model_pipeline = pipeline.from_pretrained(
                        url_or_repo,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        cache_dir=self.models_path / model_pipeline
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

def nslice(s, n, truncate=False, reverse=False):
    """Splits s into n-sized chunks, optionally reversing the chunks."""
    assert n > 0
    while len(s) >= n:
        if reverse: yield s[:n][::-1]
        else: yield s[:n]
        s = s[n:]
    if len(s) and not truncate:
        yield s

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

class CLIP_Archiver(object):
    @classmethod
    async def create(cls, civitai_token:str=None, default_model='stable-diffusion-v1-5/stable-diffusion-v1-5', models_path:str='', default_user_config:dict=None, 
        logger=None, profiles_path:str=None, return_images_and_settings:bool=False) -> None:
        """
        Create and return the CLIP_Archiver class object, initialising it with the given config.

        civitai_token - str authenticator token for the civitai api, which is required if you wish to have downloads function.
        default_model - huggingface repo id or a link to a model hosted on civitai (default 'runwayml/stable-diffusion-v1-5')
        models_path - str or pathlike object with the intended location for the models. (default '')
        default_user_config - dict of settings for users to start with. See below for the needed dict keys. (default None)
        logger - logging object (default None)
        profiles_path - str or pathlike object pointing to a json file which will be used to store user profiles.
        return_images_and_settings - bool which sets whether the images are returned in a tuple alongside the dict of settings used to make it. (default False)
        """
        self = CLIP_Archiver()
        self.change_model_queue = asyncio.Queue(maxsize=0)
        if not default_user_config:
            default_user_config = {"height": 768, "width": 768, "num_inference_steps": 22, "guidance_scale": 8.0, "scheduler": "ddim", 
            "batch_size": 1, "negative_prompt": "jpeg", "hires_fix": "False", "hires_strength": 0.75, "init_image": None, "init_strength": 0.75, "seed": None, "clip_skip": 0,
            "lora_and_embeds": None}
        self.default_user_config = default_user_config
        self.finished_generation = asyncio.Event()
        self.idle_manager = asyncio.create_task(self._idle_manager())
        self.image_queue = asyncio.Queue(maxsize=0)
        self.image_worker = asyncio.create_task(self._image_worker())
        self.logger = logger
        self.model_manager = await Model_Manager.create(
            civitai_token=civitai_token, 
            default_model=default_model, 
            models_path=models_path, 
            logger=logger
        )
        self.return_images_and_settings = return_images_and_settings
        if profiles_path:
            # Use the provided path to store user profiles.
            self.profiles_path = Path(profiles_path)
            try:
                self.user_profiles = await Async_JSON.async_load_json(self.profiles_path)
            except:
                with open(self.profiles_path, 'w') as f:
                    json.dump({}, f)
                self.user_profiles = {}
        else:
            self.profiles_path = None
            self.user_profiles = {}
        self.pipe_config = None
        self.pipeline_text2image = None

        return self

    async def __call__(self, prompt:str, batch_size:int=None, clip_skip:int=None, guidance_scale:float=None, height:int=None, hires_fix:bool=False, hires_strength:float=None, 
        init_image:str=None, init_strength:float=None, lora_and_embeds:[str]=None, model:str=None, negative_prompt:str=None, num_inference_steps:int=None, 
        preset_name:str=None, user:str='system', scheduler:str=None, seed:int=None, width:int=None) -> [Image] or ([Image], dict):
        """
        Prompt a supported txt2img model with the specified settings.

        prompt - str of text describing the desired image.
        batch_size - int amount of images to be generated together.
        clip_skip - int representing which layer of clip to skip to, which could result in more accurate images. ['SD 1', 'SD 2', 'SDXL'] 
        guidance_scale - int representing how accurate to the prompt the image will be. Lower values will lean toward the model normal output while higher values will lean towards the prompt. 
        height - int representing the height of the image in pixels.
        hires_fix - bool which sets whether the image will be first generated at the model's base resolution and then upscaled to the intented height and width. ['SD 1', 'SD 2', 'SDXL']
        hires_strength - float which is used for the img2img strength of the hires_fix.
        init_image - str of a url to an image to use as a base image to generate from. ['SD 1', 'SD 2', 'SDXL']
        init_strength - float of the img2img strength for the init_image. Lower values will result in more subtle changes. 
        lora_and_embeds - list of textual inversion embeds and LORAs with their respected weight, 'lora:1'. ['SD 1', 'SD 2', 'SDXL'] 
        model - str of which txt2img model to use for the generation.
        negative_prompt - str of text describing unwanted aspects of the image. 
        num_inference_steps - int of how many iterations the image will go through. 
        preset_name - str of the preset to save the preset under. (default '_intermediate')
        user - str of the profile to save the preset under. (default 'system')
        scheduler - str of how generation will be solved.
        seed - int representing a unique noise from which to start the generation.
        width - int representing the width of the image in pixels.
        """
        if len(self.model_manager.models) == 0:
            raise ValueError('There are no models yet!')

        # Correct args for comparison
        if clip_skip == 0:
            clip_skip = str(clip_skip)
        if hires_fix == False:
            hires_fix = str(hires_fix)
        if preset_name == '_intermediate':
            preset_name = 'intermediate'
        if not preset_name:
            preset_name = '_intermediate'

        # See if the image exists
        if init_image:
            try:
                init_image_exists = load_image(init_image)
            except:
                init_image = 'None'

        # Ensure the user's profile exists
        user_profile = self.user_profiles.setdefault(user, {'_intermediate': self.default_user_config})

        # Retrieve the preset from the user's profile or use the default
        preset = user_profile.get(preset_name, user_profile['_intermediate'])
        
        # Does the model exist?
        model_info = self.model_manager.get_model_info(model)
        if not model_info:
            preset['model'] = preset.setdefault('model', self.model_manager.default_model)
            model = preset['model']
            model_info = self.model_manager.get_model_info(model)

        # Compare the preset to the entered args and get the new settings
        settings = {
            'height':round(height/8)*8 if height else preset['height'],
            'width':round(width/8)*8 if width else preset['width'],
            'num_inference_steps':num_inference_steps if num_inference_steps else preset['num_inference_steps'],
            'guidance_scale':guidance_scale if guidance_scale else preset['guidance_scale'],
            'scheduler':scheduler.value if scheduler else preset['scheduler'],
            'batch_size':batch_size if batch_size else preset['batch_size'],
            'negative_prompt':negative_prompt if negative_prompt else preset['negative_prompt'],
            'init_image':init_image if init_image else preset['init_image'],
            'init_strength':init_strength if init_strength else preset['init_strength'],
            'hires_fix': hires_fix if hires_fix else preset['hires_fix'],
            'hires_strength': hires_strength if hires_strength else preset['hires_strength'],
            'seed':seed if seed else preset['seed'],
            'clip_skip':clip_skip if clip_skip else preset['clip_skip'],
            'lora_and_embeds':[lora_or_embed for lora_or_embed in lora_and_embeds.split(' ') if self.model_manager.get_model_info(lora_or_embed.split(':')[0])] if lora_and_embeds else preset['lora_and_embeds'],
            'model':model
        }

        # Correct args for input
        if settings['clip_skip'] == '0':
            settings['clip_skip'] = None
        if settings['hires_fix'] == 'False':
            settings['hires_fix'] = False
        if settings['init_image'] == 'None':
            settings['init_image'] = None
        if settings['seed']:
            if settings['seed'] <= -1:
                settings['seed'] = None

        # Save the settings
        user_profile[preset_name] = settings
        if not preset_name == '_intermediate':
            user_profile['_intermediate'] = settings
        if self.profiles_path:
            await Async_JSON.async_save_json(self.profiles_path, self.user_profiles)

        # Get pipeline specific settings
        settings_to_pipe = {
            'prompt':prompt,
            'height':settings['height'],
            'width':settings['width'],
            'num_inference_steps':settings['num_inference_steps'],
            'guidance_scale':settings['guidance_scale'],
            'batch_size':settings['batch_size'],
            'model': settings['model'],
            'seed': settings['seed']
        }

        if 'AuraFlow' in model_info['model_pipeline']:
            settings_to_pipe.update({'negative_prompt': settings['negative_prompt']})
        if 'Flux' in model_info['model_pipeline']:
            settings_to_pipe.update({'negative_prompt':settings['negative_prompt']})
            if settings.get('init_image'):
                settings_to_pipe.update({'init_image':settings['init_image'], 'init_strength':settings['init_strength']})
            if settings.get('hires_fix'):
                settings_to_pipe.update({'hires_fix':settings['hires_fix'], 'hires_strength':settings['hires_strength']})
            if settings.get('lora_and_embeds'):
                settings_to_pipe.update({'lora_and_embeds':settings['lora_and_embeds']})
        if model_info['model_pipeline'] in ['SD 1', 'SD 2', 'SDXL']: 
            settings_to_pipe.update({'negative_prompt':settings['negative_prompt'], 'scheduler':settings['scheduler'], 'clip_skip':settings['clip_skip']})
            if settings.get('init_image'):
                settings_to_pipe.update({'init_image':settings['init_image'], 'init_strength':settings['init_strength']})
            if settings.get('hires_fix'):
                settings_to_pipe.update({'hires_fix':settings['hires_fix'], 'hires_strength':settings['hires_strength']})
            if settings.get('lora_and_embeds'):
                settings_to_pipe.update({'lora_and_embeds':settings['lora_and_embeds']})
        if 'SD 3' in model_info['model_pipeline']:
            settings_to_pipe.update({'negative_prompt':settings['negative_prompt'], 'clip_skip':settings['clip_skip']})

        future = asyncio.Future()
        await self.image_queue.put((future, settings_to_pipe))
        return await future

    async def _change_model(self, settings_to_pipe:dict):
        """
        Informs the `_idle_manager` to change model. 
        """
        pipe_future = asyncio.Future()
        await self.change_model_queue.put((settings_to_pipe, pipe_future))
        await asyncio.sleep(1)
        return await pipe_future

    async def _idle_manager(self):
        """
        Manages the diffuser pipeline to save memory when not in use.
        """        
        while True:
            settings_to_pipe, pipe_future = await self.change_model_queue.get()
            self.finished_generation.clear()
            with torch.no_grad():
                try:
                    # need to make this is all async so it doesnt hold up the entire process #
                    pipeline_text2image = self.model_manager.build_pipeline(settings_to_pipe)
                    self.pipeline_text2image, pipe_config = pipeline_text2image()
                    self.pipeline_text2image.set_progress_bar_config(disable=True)
                    self.pipeline_text2image.enable_model_cpu_offload()
                    pipe_future.set_result(pipe_config)
                    await self.finished_generation.wait()
                    del self.pipeline_text2image
                except Exception as exception:
                    pipe_future.set_exception(exception)
                    self.pipeline_text2image = None
                    del self.pipeline_text2image
                torch.cuda.empty_cache()
                gc.collect()

    async def _image_worker(self) -> None:
        """
        This asynchronous worker loop is responsible for processing messages one by one from the FIFO image_queue.
        """
        def generate_images(pipe_config:dict) -> [Image]:
            """
            Generate an image using the specified user prompt.
            """
            if pipe_config.get('image', None):
                # Split the images into groups of 2 to save vram when doing img2img.
                images = []
                sliced_images = nslice(pipe_config['image'], 2)
                sliced_generators = nslice(pipe_config['generator'], 2)
                for image_block, generator_block in zip(sliced_images, sliced_generators):
                    pipe_config['image'] = image_block
                    pipe_config['generator'] = generator_block
                    pipe_config['num_images_per_prompt'] = len(image_block)
                    images_part = self.pipeline_text2image(**pipe_config).images
                    images.extend(images_part)
            else:
                images = self.pipeline_text2image(**pipe_config).images
            self.finished_generation.set()
            
            return images

        async def process_message(future:asyncio.Future, settings_to_pipe:dict) -> [Image] or ([Image], dict):
            """
            Prepare and send a response for the user message using their prompt.
            """
            pipe_config = await self._change_model(settings_to_pipe)
            # Diffuse!
            with torch.no_grad():
                images = await asyncio.to_thread(generate_images, pipe_config)

            # Optionally upscale with another go if the model supports img2img.
            if settings_to_pipe.get('hires_fix'):
                images = resize_and_crop_centre(images, settings_to_pipe['width'], settings_to_pipe['height'])
                pipe_config['strength'] = settings_to_pipe['hires_strength']
                settings_to_pipe.update({'hires_run':True})
                pipe_config = await self._change_model(settings_to_pipe)
                pipe_config['image'] = images

                settings_to_pipe.pop('hires_run')
                images = await asyncio.to_thread(generate_images, pipe_config)

            # Correct args for output
            if settings_to_pipe.get('clip_skip', False) == None: 
                settings_to_pipe['clip_skip'] = 0 if settings_to_pipe['clip_skip'] == None else settings_to_pipe['clip_skip']

            if self.return_images_and_settings:
                settings_to_pipe['seed'] = [settings_to_pipe['seed']+i for i in range(settings_to_pipe['batch_size'])]
                images = (images, settings_to_pipe)
            future.set_result(images)

        while True:
            ### Worker Loop ###
            try:
                future, settings_to_pipe = await self.image_queue.get()
                await process_message(future, settings_to_pipe)
            except Exception as exception:
                if self.logger:
                    self.logger.error(f"Diffuser encountered an error while generating:\n{exception}")
                future.set_exception(exception)