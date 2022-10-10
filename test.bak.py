from diffusers import VQDiffusionScheduler
import torch
from torch.nn import functional as F

from orig_scheduler import OrigScheduler

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

    
num_embed = 3
content_seq_len = 3

scheduler = VQDiffusionScheduler(num_embed=num_embed)
orig_scheduler = OrigScheduler(num_classes=num_embed, content_seq_len=content_seq_len)


t = torch.tensor([3])

x_t = torch.tensor([
    [2, 0, 1]
])
log_x_t = index_to_log_onehot(x_t, num_embed)

# tensor([[[-2.9957, -0.3567, -0.2231],
#          [-0.0513, -1.2040, -1.6094]]])
log_x_0 = torch.tensor([
    [
        [0.05, 0.7, 0.8],
        [0.95, 0.3, 0.2]
    ]
]).log().clamp(min=-70)

res = scheduler.q_posterior(log_x_0=log_x_0, x_t=x_t, t=t)

print(res)
print(res.argmax(1))

res_orig = orig_scheduler.q_posterior(log_x_0, log_x_t, t)

print(res_orig)
print(res_orig.argmax(1))

from diffusers import VQDiffusionPipeline
import PIL
import numpy as np
import torch

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

pipeline = VQDiffusionPipeline.from_pretrained('/content/vq-diffusion-diffusers-dump')

device = 'cuda'

pipeline = pipeline.to(device)

vqvae = pipeline.vqvae

input_file_name = "/content/cat.jpg"

image = PIL.Image.open(input_file_name).convert("RGB")
image = preprocess_image(image).to(device)

with torch.no_grad():
    encoded = vqvae.encode(image).latents
    # Original vq-diffusion uses the min encoding indices
    _, _, (_, _, encoded_min_encoding_indices) = vqvae.quantize(encoded)

###############


self = pipeline

prompt = "horse"

text_inputs = self.tokenizer(
    prompt,
    padding="max_length",
    max_length=self.tokenizer.model_max_length,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids

text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

batch_size = 1
latents_shape = (batch_size, self.transformer.num_embed)
mask_class = self.transformer.num_embed - 1
latents = torch.full(latents_shape, mask_class).to(self.device)

self.scheduler.set_timesteps(100)

timesteps_tensor = self.scheduler.timesteps.to(self.device)

self.progress_bar(timesteps_tensor)

# pipeline("horse")
