import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
import ipdb
from diffusers import DiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...image_processor import VaeImageProcessor
from ...models import UVit2DModel, VQModel
from ...schedulers import OpenMuseScheduler
from ...utils import replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        images (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`[List[bool]]`):
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    # warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    # warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="border")
    warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="zeros")
    return warped


def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype):
    """
    Create translation motion field

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        device: device
        dtype: dtype

    Returns:

    """
    seq_length = len(frame_ids)
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    for fr_idx in range(seq_length):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx])
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx])
    return reference_flow


def create_motion_field_and_warp_latents(motion_field_strength_x, motion_field_strength_y, frame_ids, latents):
    """
    Creates translation motion and warps the latents accordingly

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        latents: latent codes of frames

    Returns:
        warped_latents: warped latents
    """
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        frame_ids=frame_ids,
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents

class OpenMuseTextToVideoZeroPipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: UVit2DModel
    scheduler: OpenMuseScheduler

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: UVit2DModel,
        scheduler: OpenMuseScheduler,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    def backward_loop(
        self,
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        callback,
        callback_steps,
        num_warmup_steps,
        extra_step_kwargs,
        cross_attention_kwargs=None,
        muse_prompt_embeds=None,
        muse_micro_conds=None,
        muse_encoder_hidden_states=None,
        muse_generator=None,
        muse=False
    ):
        """
        Perform backward process given list of time steps.

        Args:
            latents:
                Latents at time timesteps[0].
            timesteps:
                Time steps along which to perform backward process.
            prompt_embeds:
                Pre-generated text embeddings.
            guidance_scale:
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            extra_step_kwargs:
                Extra_step_kwargs.
            cross_attention_kwargs:
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            num_warmup_steps:
                number of warmup steps.

        Returns:
            latents:
                Latents of backward process output at time timesteps[-1].
        """
        assert muse
        assert muse_prompt_embeds is not None
        assert muse_micro_conds is not None
        assert muse_encoder_hidden_states is not None
        assert muse_generator is not None
        self = self.muse_pipe
        extra_step_kwargs = {"generator": muse_generator}

        do_classifier_free_guidance = guidance_scale > 1.0

        num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order

        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    micro_conds=muse_micro_conds,
                    cond_embeds=muse_prompt_embeds,
                    encoder_hidden_states=muse_encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        return latents.clone().detach()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int] = 8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        frame_ids: Optional[List[int]] = None,
        muse_guidance_scale=10.0,
        muse_num_inference_steps=12,
        muse_t1=9,
        muse_generator=None,
        muse_latents=None,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            video_length (`int`, *optional*, defaults to 8):
                The number of generated video frames.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in video generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"numpy"`):
                The output format of the generated video. Choose between `"latent"` and `"numpy"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput`] instead of
                a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            motion_field_strength_x (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along x-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            motion_field_strength_y (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along y-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            t0 (`int`, *optional*, defaults to 44):
                Timestep t0. Should be in the range [0, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            t1 (`int`, *optional*, defaults to 47):
                Timestep t0. Should be in the range [t0 + 1, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            frame_ids (`List[int]`, *optional*):
                Indexes of the frames that are being generated. This is used when generating longer videos
                chunk-by-chunk.

        Returns:
            [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput`]:
                The output contains a `ndarray` of the generated video, when `output_type` != `"latent"`, otherwise a
                latent code of generated videos and a list of `bool`s indicating whether the corresponding generated
                video contains "not-safe-for-work" (nsfw) content..
        """
        assert video_length > 0
        if frame_ids is None:
            frame_ids = list(range(video_length))
        assert len(frame_ids) == video_length

        assert num_videos_per_prompt == 1

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        device = self._execution_device

        if height is None:
            height = self.muse_pipe.transformer.config.sample_size * self.muse_pipe.vae_scale_factor

        if width is None:
            width = self.muse_pipe.transformer.config.sample_size * self.muse_pipe.vae_scale_factor

        self.muse_pipe.scheduler.set_timesteps(muse_num_inference_steps, device=device)
        muse_timesteps = self.muse_pipe.scheduler.timesteps

        # Prepare extra step kwargs.
        extra_step_kwargs = {}

        if muse_latents is None:
            muse_latents = torch.full(
                (1, 1024), self.muse_pipe.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
            )

        muse_micro_conds, muse_prompt_embeds, muse_encoder_hidden_states = self.muse_setup(prompt, muse_guidance_scale)

        muse_x_1_t1 = self.backward_loop(
            muse=True,
            # timesteps=muse_timesteps[: -muse_t1 - 1],
            timesteps=muse_t1,
            prompt_embeds=None,
            muse_prompt_embeds=muse_prompt_embeds,
            muse_micro_conds=muse_micro_conds,
            muse_encoder_hidden_states=muse_encoder_hidden_states,
            muse_generator=muse_generator,
            latents=muse_latents,
            guidance_scale=muse_guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=0,
        )

        muse_x_1_t0 = muse_x_1_t1

        # Propagate first frame latents at time T_0 to remaining frames
        muse_x_2k_t0 = muse_x_1_t0.repeat(video_length - 1, 1)

        # de-quantize
        muse_x_2k_t0 = self.muse_pipe.vqvae.quantize.get_codebook_entry(muse_x_2k_t0, (muse_x_2k_t0.shape[0], height // self.muse_pipe.vae_scale_factor, width // self.muse_pipe.vae_scale_factor, self.muse_pipe.vqvae.latent_channels))

        # warp
        muse_x_2k_t0 = create_motion_field_and_warp_latents(
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            latents=muse_x_2k_t0,
            frame_ids=frame_ids[1:],
        )

        # replace padding with mask
        padded_indices = (muse_x_2k_t0 == 0).all(dim=1, keepdim=True)
        masked_embedding = self.muse_pipe.vqvae.quantize.embedding.weight[-1]
        n_channels = muse_x_2k_t0.shape[1]
        muse_x_2k_t0 = muse_x_2k_t0.permute(0, 2, 3, 1)
        muse_x_2k_t0[
            padded_indices.repeat(1, n_channels, 1, 1).permute(0, 2, 3, 1)
        ] = masked_embedding.repeat(padded_indices.sum())
        muse_x_2k_t0 = muse_x_2k_t0.permute(0, 3, 1, 2)

        # quantize
        muse_x_2k_t0 = self.muse_pipe.vqvae.quantize(muse_x_2k_t0)[2][2].reshape(muse_x_2k_t0.shape[0], -1)

        muse_x_2k_t1 = muse_x_2k_t0

        muse_x_1k_t1 = torch.cat([muse_x_1_t1, muse_x_2k_t1])

        b, l = muse_prompt_embeds.size()
        muse_prompt_embeds = muse_prompt_embeds[:, None, :].repeat(1, video_length, 1).reshape(b * video_length, l)

        b, l, d = muse_encoder_hidden_states.size()
        muse_encoder_hidden_states = muse_encoder_hidden_states[:, None, :, :].repeat(1, video_length, 1, 1).reshape(b * video_length, l, d)

        b, l = muse_micro_conds.size()
        muse_micro_conds = muse_micro_conds[:, None, :].repeat(1, video_length, 1).reshape(b * video_length, l)

        muse_x_1k_0 = muse_x_1k_t1

        muse_latents = muse_x_1k_0

        muse_image = self.muse_pipe.vqvae.decode(
            muse_latents,
            force_not_quantize=True,
            shape=(
                muse_latents.shape[0],
                height // self.muse_pipe.vae_scale_factor,
                width // self.muse_pipe.vae_scale_factor,
                self.muse_pipe.vqvae.config.latent_channels,
            ),
        ).sample
        muse_image = self.muse_pipe.image_processor.postprocess(muse_image, output_type)

        image = None
        has_nsfw_concept = None

        if not return_dict:
            return (image, has_nsfw_concept, muse_image, muse_latents)

        return TextToVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    def muse_setup(it, prompt, muse_guidance_scale):
        self = it.muse_pipe
        guidance_scale = muse_guidance_scale
        num_images_per_prompt = 1
        negative_prompt_embeds = None
        negative_prompt = None

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        height = self.transformer.config.sample_size * self.vae_scale_factor

        width = self.transformer.config.sample_size * self.vae_scale_factor

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids.to(self._execution_device)

        outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.text_embeds
        encoder_hidden_states = outputs.hidden_states[-2]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                input_ids = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)

                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                negative_prompt_embeds = outputs.text_embeds
                negative_encoder_hidden_states = outputs.hidden_states[-2]

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        micro_conds = torch.tensor(
            [height, width, 0, 0, 6], device=self._execution_device, dtype=encoder_hidden_states.dtype
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        return micro_conds, prompt_embeds, encoder_hidden_states