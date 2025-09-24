def qwen_image(guidance_scale:float, height:int, num_inference_steps:int, num_images_per_prompt:int, prompt:str, width:int, init_image:str=None, init_strength:float=None, lora_and_embeds:list=None, 
            model_info:dict=model_info, negative_prompt:str=None, hires_fix:bool=None, hires_strength:float=None, **kwargs):
            """
            Build a pipeline for a QwenImage model.
            """
            torch_dtype = torch.bfloat16
            model_pipeline = model_info['model_pipeline'] 
            if init_image:
                model_pipeline += "Img2Img"
            pipeline = self.get_pipeline(model_pipeline)

            # encode the prompt
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch_dtype,
                transformer=None,
                vae=None
            ).to('cuda')

            prompt_embeds, prompt_embeds_mask = pipeline_text2image.encode_prompt(prompt=prompt, num_images_per_prompt=num_images_per_prompt, max_sequence_length=1024, device='cuda')
            negative_prompt_embeds, negative_prompt_embeds_mask = pipeline_text2image.encode_prompt(prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt, max_sequence_length=1024, device='cuda')
            del pipeline_text2image

            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                text_encoder=None,
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

            # guidance_scale is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. 
            # To enable traditional classifier-free guidance, please pass true_cfg_scale > 1.0 and negative_prompt
            # https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage#diffusers.QwenImagePipeline.__call__
            pipe_config['true_cfg_scale'] = guidance_scale
            pipe_config['guidance_scale'] = guidance_scale

            pipe_config['height'] = height
            pipe_config['num_inference_steps'] = num_inference_steps
            pipe_config['num_images_per_prompt'] = num_images_per_prompt
            pipe_config['negative_prompt_embeds'] = negative_prompt_embeds
            pipe_config['negative_prompt_embeds_mask'] = negative_prompt_embeds_mask
            pipe_config['prompt_embeds'] = prompt_embeds
            pipe_config['prompt_embeds_mask'] = prompt_embeds_mask
            pipe_config['width'] = width

            return pipeline_text2image, pipe_config
