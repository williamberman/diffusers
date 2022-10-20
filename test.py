from diffusers import VQDiffusionPipeline

device = 'cuda'

pipeline = VQDiffusionPipeline.from_pretrained('/content/vq-diffusion-diffusers-dump').to(device)

image = pipeline("horse").images[0]

image.save("/content/out.jpg")
