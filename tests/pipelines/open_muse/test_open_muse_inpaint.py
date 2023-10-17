# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import OpenMuseInpaintPipeline, OpenMuseScheduler, UVit2DModel, VQModel
from diffusers.utils import load_image
from diffusers.utils.import_utils import is_flash_attn_available
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class OpenMuseInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = OpenMuseInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS - {"width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
    }

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = UVit2DModel(
            hidden_size=32,
            use_bias=False,
            hidden_dropout=0.0,
            cond_embed_dim=32,
            micro_cond_encode_dim=2,
            micro_cond_embed_dim=10,
            encoder_hidden_size=32,
            vocab_size=32,
            codebook_size=32,
            in_channels=32,
            block_out_channels=32,
            num_res_blocks=1,
            downsample=True,
            upsample=True,
            block_num_heads=1,
            num_hidden_layers=1,
            num_attention_heads=1,
            attention_dropout=0.0,
            intermediate_size=32,
            layer_norm_eps=1e-06,
            ln_elementwise_affine=True,
        )
        scheduler = OpenMuseScheduler(mask_token_id=31)
        torch.manual_seed(0)
        vqvae = VQModel(
            act_fn="silu",
            block_out_channels=[32],
            down_block_types=[
                "DownEncoderBlock2D",
            ],
            in_channels=3,
            latent_channels=32,
            layers_per_block=2,
            norm_num_groups=32,
            num_vq_embeddings=32,
            out_channels=3,
            sample_size=32,
            up_block_types=[
                "UpDecoderBlock2D",
            ],
            mid_block_add_attention=False,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=32,
        )
        text_encoder = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "transformer": transformer,
            "scheduler": scheduler,
            "vqvae": vqvae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        image = torch.full((1, 3, 4, 4), 1.0, dtype=torch.float32, device=device)
        mask_image = torch.full((1, 1, 4, 4), 1.0, dtype=torch.float32, device=device)
        mask_image[0, 0, 0, 0] = 0
        mask_image[0, 0, 0, 1] = 0
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
            "image": image,
            "mask_image": mask_image,
        }
        return inputs

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes, batch_generator=False)

    @unittest.skip("Open muse does not support lists of generators")
    def test_inference_batch_single_identical(self):
        ...

    @require_torch_gpu
    @unittest.skipIf(not is_flash_attn_available(), "fused rms norm is not installed")
    def test_fused_rms_norm(self):
        pipeline = self.pipeline_class(**self.get_dummy_components())
        pipeline.to(torch_device)

        out_no_fused_rms_norm = pipeline(**self.get_dummy_inputs(torch_device)).images

        pipeline.enable_fused_rms_norm()

        out_fused_rms_norm = pipeline(**self.get_dummy_inputs(torch_device)).images

        assert np.abs(out_no_fused_rms_norm - out_fused_rms_norm).max() < 3e-3


@slow
@require_torch_gpu
class OpenMuseInpaintPipelineSlowTests(unittest.TestCase):
    def test_open_muse_256(self):
        pipe = OpenMuseInpaintPipeline.from_pretrained("openMUSE/diffusers-pipeline-256")  # TODO change
        pipe.to(torch_device)

        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((256, 256))
            .convert("RGB")
        )

        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((256, 256))
            .convert("L")
        )

        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0042, -0.0017, -0.0095, -0.0010, 0.0075, -0.0022, 0.0040, 0.0239, -0.0053])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_open_muse_512(self):
        pipe = OpenMuseInpaintPipeline.from_pretrained("openMUSE/diffusers-pipeline")  # TODO change
        pipe.to(torch_device)

        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((512, 512))
            .convert("RGB")
        )

        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((512, 512))
            .convert("L")
        )

        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([-0.0203, -0.0261, -0.0308, -0.0221, -0.0187, -0.0294, -0.0187, -0.0061, -0.0343])
        assert np.abs(image_slice - expected_slice).max() < 3e-3
