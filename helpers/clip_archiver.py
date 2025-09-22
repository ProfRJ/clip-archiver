import asyncio
import gc
import io
import json
import time
import torch

from diffusers.utils import load_image 
from helpers import Async_JSON
from helpers.model_manager import Model_Manager
from pathlib import Path
from PIL import Image


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
        save_as_4bit:bool=True, save_as_8bit:bool=False, logger=None, profiles_path:str=None, return_images_and_settings:bool=False) -> None:
        """
        Create and return the CLIP_Archiver class object, initialising it with the given config.

        civitai_token - str authenticator token for the civitai api, which is required if you wish to have downloads function.
        default_model - str with huggingface repo id or a link to a model hosted on civitai (default 'runwayml/stable-diffusion-v1-5')
        models_path - str or pathlike object with the intended location for the models. (default '')
        default_user_config - dict of settings for users to start with. See below for the needed dict keys. (default None)
        logger - logging object (default None)
        profiles_path - str or pathlike object pointing to a json file which will be used to store user profiles.
        return_images_and_settings - bool which sets whether the images are returned in a tuple alongside the dict of settings used to make it. (default False)
        """
        self = CLIP_Archiver()
        self.change_model_queue = asyncio.Queue(maxsize=0)
        if not default_user_config:
            default_user_config = {"height": 768, "width": 768, "num_images_per_prompt":1, "num_inference_steps": 22, "guidance_scale": 8.0, "scheduler": "ddim", 
            "negative_prompt": "jpeg", "true_cfg_scale":1.0, "hires_fix": "False", "hires_strength": 0.75, "init_image": None, "init_strength": 0.75, "seed": -1, "clip_skip": 0,
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
            save_as_4bit=save_as_4bit,
            save_as_8bit= save_as_8bit,
            logger=logger, 
            models_path=models_path
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

    async def __call__(self, prompt:str, clip_skip:int=None, guidance_scale:float=None, height:int=None, hires_fix:bool=False, hires_strength:float=None, init_image:str=None, 
        init_strength:float=None, lora_and_embeds:[str]=None, model:str=None, negative_prompt:str=None, num_images_per_prompt:int=None, num_inference_steps:int=None, 
        preset_name:str='_intermediate', true_cfg_scale:float=None, user:str='system', scheduler:str=None, seed:int=None, width:int=None) -> [Image] or ([Image], dict):
        """
        Prompt a supported txt2img model with the specified settings.

        prompt - str of text describing the desired image.
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
        num_images_per_prompt - int amount of images to be generated together.
        num_inference_steps - int of how many iterations the image will go through. 
        preset_name - str of the preset to save the preset under. (default '_intermediate')
        true_cfg_scale - float that is used in place of guidance_scale in some models.
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
            'num_images_per_prompt': num_images_per_prompt if num_images_per_prompt else preset['num_images_per_prompt'],
            'num_inference_steps':num_inference_steps if num_inference_steps else preset['num_inference_steps'],
            'guidance_scale':guidance_scale if guidance_scale else preset['guidance_scale'],
            'scheduler':scheduler.value if scheduler else preset['scheduler'],
            'negative_prompt':negative_prompt if negative_prompt else preset['negative_prompt'],
            'true_cfg_scale':true_cfg_scale if true_cfg_scale else preset['true_cfg_scale'],
            'init_image':init_image if init_image else preset['init_image'],
            'init_strength':init_strength if init_strength else preset['init_strength'],
            'hires_fix': hires_fix if hires_fix else preset['hires_fix'],
            'hires_strength': hires_strength if hires_strength else preset['hires_strength'],
            'seed':seed if seed else preset['seed'],
            'clip_skip':clip_skip if clip_skip else preset['clip_skip'],
            'lora_and_embeds':[lora_or_embed for lora_or_embed in lora_and_embeds.split(' ') if self.model_manager.get_model_info(lora_or_embed.split(':')[0])] if lora_and_embeds else preset['lora_and_embeds'],
            'model':model
        }

        # Save the settings
        user_profile[preset_name] = settings.copy()
        if not preset_name == '_intermediate':
            user_profile['_intermediate'] = settings
        if self.profiles_path:
            await Async_JSON.async_save_json(self.profiles_path, self.user_profiles)

        # Correct args for input
        settings['prompt'] = prompt
        if settings['clip_skip'] == '0':
            settings['clip_skip'] = None
        if settings['hires_fix'] == 'False':
            settings['hires_fix'] = False
        if settings['init_image'] == 'None':
            settings['init_image'] = None
        if settings['seed'] == None or -1:
            settings['seed'] = int(time.time())
        if len(settings['lora_and_embeds']) == 0:
            settings.pop('lora_and_embeds')

        future = asyncio.Future()
        await self.image_queue.put((future, settings))
        return await future

    async def _change_model(self, settings:dict):
        """
        Informs the `_idle_manager` to change model. 
        """
        pipe_future = asyncio.Future()
        await self.change_model_queue.put((settings, pipe_future))
        return await pipe_future

    async def _idle_manager(self):
        """
        Manages the diffuser pipeline to save memory when not in use.
        """        
        while True:
            settings, pipe_future = await self.change_model_queue.get()
            self.finished_generation.clear()
            try:
                with torch.no_grad():
                    pipeline_text2image = self.model_manager.build_pipeline(settings)
                    self.pipeline_text2image, pipe_config = pipeline_text2image(**settings)
                    self.pipeline_text2image.set_progress_bar_config(disable=True)
                    self.pipeline_text2image.enable_model_cpu_offload()
                pipe_future.set_result(pipe_config)
                await self.finished_generation.wait()
            except Exception as exception:
                pipe_future.set_exception(exception)
            finally:
                if self.pipeline_text2image:
                    del self.pipeline_text2image
                    self.pipeline_text2image = None
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

        async def process_message(future:asyncio.Future, settings:dict) -> [Image] or ([Image], dict):
            """
            Prepare and send a response for the user message using their settings.
            """
            pipe_config = await self._change_model(settings)
            # Diffuse!
            with torch.no_grad():
                images = await asyncio.to_thread(generate_images, pipe_config)

            # Optionally upscale with another go if the model supports img2img.
            if settings.get('hires_fix'):
                images = resize_and_crop_centre(images, settings['width'], settings['height'])
                hires_settings = settings.copy()
                hires_settings['image'] = images
                hires_settings['hires_fix'] = False    
                hires_settings['strength'] = settings['hires_strength']
                hires_pipe_config = await self._change_model(hires_settings)

                with torch.no_grad():
                    images = await asyncio.to_thread(generate_images, hires_pipe_config)

            # Correct args for output
            if settings.get('clip_skip', False) == None: 
                settings['clip_skip'] = 0 if settings['clip_skip'] == None else settings['clip_skip']

            # Optionally make the result a tuple to add a dictionary of settings used to generate each image.
            if self.return_images_and_settings:
                settings['seed'] = [settings['seed']+i for i in range(settings['num_images_per_prompt'])]
                images = (images, settings)

            future.set_result(images)

        while True:
            ### Worker Loop ###
            try:
                future, settings = await self.image_queue.get()
                await process_message(future, settings)
            except Exception as exception:
                if self.logger:
                    self.logger.error(f"Diffuser encountered an error while generating:\n{exception}")
                future.set_exception(exception)
