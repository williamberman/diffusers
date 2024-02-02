import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import os
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.unet.requires_grad_(True)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, output_type="latent", timesteps=np.array([500, 250]), start_ts=0, end_ts=1, num_inference_steps=1, generator=torch.Generator('cpu').manual_seed(0), guidance_scale=1).images[0]
print(f"manual step 1 image out: {image.abs().sum()}")
image.sum().backward(retain_graph=True)
step_1_grad = pipe.unet.conv_in.weight.grad.clone().detach()
print(f"manual step 1 grad: {step_1_grad.sum()}")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, output_type="latent", latents=image.unsqueeze(0), timesteps=np.array([500, 250]), start_ts=1, end_ts=2, skip_prepare_latents=True, num_inference_steps=1, generator=torch.Generator('cpu').manual_seed(0), guidance_scale=1).images[0]
print(f"manual step 2 image out: {image.abs().sum()}")
image[0].sum().backward(retain_graph=True)
print(f"manual step 2 grad: {(pipe.unet.conv_in.weight.grad - step_1_grad).sum()}") # first step gradient has already been calculated, so we're accumulating it twice by calling backward again. Subtract one copy of the first step gradient to get the true two step gradient

pipe.unet.zero_grad()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, output_type="latent", timesteps=np.array([500, 250]), start_ts=0, end_ts=2, num_inference_steps=2, generator=torch.Generator('cpu').manual_seed(0), guidance_scale=1).images[0]
print(image.abs().sum())
image[0].sum().backward(retain_graph=True)
print(f"step 1 and 2 image out: {image.abs().sum()}")
print(f"step 1 and 2 grad {pipe.unet.conv_in.weight.grad.sum()}")
