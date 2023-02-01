import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from ...models import AutoencoderKL, NoiseAugmentor, PriorTransformer, UNet2DConditionModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from ...utils import is_accelerate_available, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableUnCLIPPipeline(DiffusionPipeline):
    # prior components
    prior_tokenizer: CLIPTokenizer
    prior_text_encoder: CLIPTextModelWithProjection
    prior: PriorTransformer
    prior_scheduler: DDPMScheduler

    # regular denoising components
    noise_augmentor: NoiseAugmentor
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler

    vae: AutoencoderKL

    def __init__(
        self,
        # prior components
        prior_tokenizer: CLIPTokenizer,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior: PriorTransformer,
        prior_scheduler: DDPMScheduler,
        # regular denoising components
        noise_augmentor: NoiseAugmentor,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()

        self.register_modules(
            prior_tokenizer=prior_tokenizer,
            prior_text_encoder=prior_text_encoder,
            prior=prior,
            prior_scheduler=prior_scheduler,
            noise_augmentor=noise_augmentor,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        # TODO: self.prior.post_process_latents is not covered by the offload hooks, so it fails if added to the list
        models = [
            self.prior_text_encoder,
            self.noise_augmentor,
            self.text_encoder,
            self.unet,
            self.vae,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline._encode_prompt with _encode_prompt->_encode_prior_prompt, tokenizer->prior_tokenizer, text_encoder->prior_text_encoder
    def _encode_prior_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        if text_model_output is None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
            text_inputs = self.prior_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.prior_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_mask = text_inputs.attention_mask.bool().to(device)

            untruncated_ids = self.prior_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.prior_tokenizer.batch_decode(
                    untruncated_ids[:, self.prior_tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.prior_tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.prior_tokenizer.model_max_length]

            prior_text_encoder_output = self.prior_text_encoder(text_input_ids.to(device))

            prompt_embeds = prior_text_encoder_output.text_embeds
            prior_text_encoder_hidden_states = prior_text_encoder_output.last_hidden_state

        else:
            batch_size = text_model_output[0].shape[0]
            prompt_embeds, prior_text_encoder_hidden_states = text_model_output[0], text_model_output[1]
            text_mask = text_attention_mask

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prior_text_encoder_hidden_states = prior_text_encoder_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            uncond_input = self.prior_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.prior_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_prior_text_encoder_output = self.prior_text_encoder(
                uncond_input.input_ids.to(device)
            )

            negative_prompt_embeds = negative_prompt_embeds_prior_text_encoder_output.text_embeds
            uncond_prior_text_encoder_hidden_states = (
                negative_prompt_embeds_prior_text_encoder_output.last_hidden_state
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_prior_text_encoder_hidden_states.shape[1]
            uncond_prior_text_encoder_hidden_states = uncond_prior_text_encoder_hidden_states.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_prior_text_encoder_hidden_states = uncond_prior_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prior_text_encoder_hidden_states = torch.cat(
                [uncond_prior_text_encoder_hidden_states, prior_text_encoder_hidden_states]
            )

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, prior_text_encoder_hidden_states, text_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prior_guidance_scale: float = 4.0,
        guidance_scale: float = 10,
        noise_level: int = 0,
        eta: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        prior_num_inference_steps: int = 25,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prior_latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # TODO check_inputs

        if isinstance(prompt, str):
            batch_size = 1
        else:
            # prompt is list
            batch_size = len(prompt)

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        prior_do_classifier_free_guidance = prior_guidance_scale > 1.0

        # 3. Encode input prompt
        prior_prompt_embeds, prior_text_encoder_hidden_states, prior_text_mask = self._encode_prior_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=prior_do_classifier_free_guidance,
        )

        # 4. Prepare prior timesteps
        self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        # 5. Prepare prior latent variables
        embedding_dim = self.prior.config.embedding_dim
        prior_latents = self.prepare_latents(
            (batch_size, embedding_dim),
            prior_prompt_embeds.dtype,
            device,
            generator,
            prior_latents,
            self.prior_scheduler,
        )

        # 6. Prior denoising loop
        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([prior_latents] * 2) if prior_do_classifier_free_guidance else prior_latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prior_prompt_embeds,
                encoder_hidden_states=prior_text_encoder_hidden_states,
                attention_mask=prior_text_mask,
            ).predicted_image_embedding

            if prior_do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
            ).prev_sample

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeds = prior_latents

        # done prior

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 7. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 8. Prepare image embeddings
        image_embeds = self.noise_augmentor(image_embeds, noise_level=noise_level, generator=generator)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        # 9. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 10. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = self.prepare_latents(
            shape=shape,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            scheduler=self.scheduler,
        )

        # 11. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 12. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 13. Post-processing
        image = self.decode_latents(latents)

        # 14. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
