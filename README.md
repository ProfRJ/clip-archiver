# CLIP-Archiver
An asynchronous interface for some select txt2img diffusion models using the [🤗 Diffusers](https://huggingface.co/docs/diffusers/index) library. Makes use of quantization and pre-encoding prompts for large models, and implements hires_fix for pipelines that support img2img. This reduces the vram needed to run large models like Stable Diffusion 3, Flux, and Auraflow, while also making the older models more competitive.

![showcase of image generation](clip_archiver_showcase.png)

## Installation:
- Create a python venv with an optional prompt: \
`python -m clip-archiver .venv`
- Enter the python venv: \
Linux: `source .venv/bin/activate` \
Windows: `.venv\Scripts\activate.bat` 
- Install required pip dependencies: \
`pip install accelerate peft protobuf pytorch_lightning requests sentencepiece transformers` \
`pip install git+https://github.com/huggingface/diffusers`

## Basic Usage:
- Create the Object: 
```
clip_archiver = await CLIP_Archiver.create(
    default_model='stable-diffusion-v1-5/stable-diffusion-v1-5', # huggingface repos or civitai links are accepted.
    models_path='models/path', # Can be a full path, or a name for the subfolder
    return_images_and_settings=True
)
```
- Call the Object:

To generate an image, send a prompt to the object. Since we set `return_images_and_settings`, `response` will be a tuple containing a list of the images generated, as well as the settings used to generate it.
```
response = await clip_archiver(prompt='Visions of Chaos by Greg Rutkowski')
images, settings = response[0], response[1]
```
Beyond that, you can also include other arguments:

| Parameter            | Type   | Description                                                                                       |
|----------------------|--------|---------------------------------------------------------------------------------------------------|
| `prompt`             | str    | Text describing the desired image.                                                                |
| `batch_size`         | int    | Amount of images to be generated together.                                                        |
| `clip_skip`*         | int    | Represents which layer of CLIP to skip to, potentially resulting in more accurate images.         |
| `guidance_scale`     | int    | Represents how accurate to the prompt the image will be; lower values lean toward normal output.  |
| `height`             | int    | Height of the image in pixels.                                                                    |
| `hires_fix`*         | bool   | Sets whether or not the image will be generated at the model's base resolution and then upscaled. |
| `hires_strength`*    | float  | Used for the img2img strength of the `hires_fix`.                                                 |
| `init_image`*        | str    | URL to an image used as a base for generation.                                                    |
| `init_strength`*     | float  | Img2img strength for the `init_image`; lower values result in more subtle changes.                |
| `lora_and_embeds`*   | list   | List of textual inversion embeds and LORAs with their respective weights (e.g., `lora:1`).        |
| `model`              | str    | The txt2img model to use for generation.                                                          |
| `negative_prompt`    | str    | Text describing unwanted aspects of the image.                                                    |
| `num_inference_steps`| int    | Number of iterations the image will go through.                                                   |
| `preset_name`        | str    | Preset name to save the preset under (default: `_intermediate`).                                  |
| `user`               | str    | Profile to save the preset under (default: `system`).                                             |
| `scheduler`*         | str    | Method of how generation will be solved.                                                          |
| `seed`               | int    | Unique noise from which to start the generation.                                                  |
| `width`              | int    | Width of the image in pixels.                                                                     |

**please note some arguments might not be available for a model's pipeline* 

## Adding Models:
On the first run, the specified model_path will be created and populated with subfolders for each accepted pipeline. So far, `AuraFlow`, `Flux.1`, `Stable Diffusion 1`, `Stable Diffusion 2`, `Stable Diffusion 3`, `Stable Diffusion XL` are supported. New models will be added on startup if they are not added manually.

To download diffuser models from huggingface, or download checkpoints from civitai, you can provide the object's `model_manager` with the link. This will download the model, perform any neccessary conversions, and save the model to the pipeline's subfolder in `models_path`.
```
model_dict = await clip_archiver.model_manager(repo/id or civitai.link, model_owner='system')
# Once added, you will be able to see it in; 
clip_archiver.model_managers.models
# Or in a more human readable format.
clip_archiver.model_manager.list_models()
```
From here, the model name can used as an argument in image generation. If you wish to remove a model you can use:
```
await clip_archiver.model_manager.remove_model(model_name, model_owner='system')    
```
This is assuming this is a small scope. If other users are involved you can pass `model_owner` to model download and removal.
