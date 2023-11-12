import torch
from diffusers.pipelines.open_muse.pipeline_open_muse_text_to_video_zero import OpenMuseTextToVideoZeroPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

model_id = "../muse-512-finetuned-convert"
pipe = OpenMuseTextToVideoZeroPipeline.from_pretrained(model_id).to("cuda")
pipe.transformer.set_attn_processor(AttnProcessor())

prompt = "a cowboy riding a horse with a city in the background"

motion_field_strength_x = -15
seed = 5

muse_image = pipe(
    prompt=prompt, 
    video_length=17,
    motion_field_strength_y=0, 
    motion_field_strength_x=-15,
    generator=torch.Generator('cuda').manual_seed(1),
    return_dict=False,
)[0]
muse_image[0].save(f"muse_text_to_video_incremental_2_motion_field_x_{motion_field_strength_x}_seed_{seed}.gif", save_all=True, append_images=muse_image[1:], duration=150, loop=0)
