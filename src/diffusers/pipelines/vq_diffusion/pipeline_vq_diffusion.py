from ...pipeline_utils import DiffusionPipeline
from diffusers import VQModel

import numpy as np
import torch
import PIL

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

# This class is a placeholder and does not have the full VQ-diffusion pipeline built out yet
#
# NOTE: In VQ-Diffusion, the VQVAE trained on the ITHQ dataset uses an EMA variant of the vector quantizer
# in diffusers. The EMA variant uses EMA's to update the codebook during training but acts the same as the
# usual vector quantizer during inference. The VQDiffusion pipeline uses the non-ema vector quantizer during
# inference. If diffusers is to support training, the EMA vector quantizer could be implemented. For more
# information on EMA Vector quantizers, see https://arxiv.org/abs/1711.00937.
class VQDiffusionPipeline(DiffusionPipeline):

    vqvae: VQModel

    def __init__(self, vqvae: VQModel):
        super().__init__()
        self.register_modules(vqvae=vqvae)

    @torch.no_grad()
    def encode(self, image):
        image = preprocess_image(image)
        encoded = self.vqvae.encode(image)
        return encoded.latents

    @torch.no_grad()
    def decode(self, encoded_image):
        image = self.vqvae.decode(encoded_image).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        return image
