import torch
# from diffusers import TextToVideoZeroPipeline
from diffusers.pipelines.open_muse.pipeline_open_muse_text_to_video_zero import OpenMuseTextToVideoZeroPipeline
from diffusers.models.attention_processor import AttnProcessor
from diffusers import OpenMusePipeline
from PIL import Image
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = OpenMuseTextToVideoZeroPipeline.from_pretrained(model_id, safety_checker=None).to("cuda")
# pipe.vae.set_attn_processor(AttnProcessor())

# muse_pipe = OpenMusePipeline.from_pretrained("../muse-512-finetuned-convert")
# muse_pipe.to('cuda')
# muse_pipe.transformer.set_attn_processor(AttnProcessor())
# muse_pipe.transformer.set_attn_processor(CrossFrameAttnProcessor(enabled=False))

model_id = "../muse-512-finetuned-convert"
pipe = OpenMuseTextToVideoZeroPipeline.from_pretrained(model_id).to("cuda")
pipe.transformer.set_attn_processor(AttnProcessor())
muse_pipe = pipe
pipe.muse_pipe = muse_pipe


embedding_layer = muse_pipe.vqvae.quantize.embedding
n_new_embeddings = muse_pipe.scheduler.config.mask_token_id - embedding_layer.num_embeddings + 1
new_embeddings = torch.randn(n_new_embeddings, embedding_layer.embedding_dim, device='cuda')
extended_weight = torch.cat([embedding_layer.weight, new_embeddings], 0)
embedding_layer.num_embeddings += n_new_embeddings
embedding_layer.weight = torch.nn.Parameter(extended_weight)

pipe.muse_pipe = muse_pipe
pipe.do_sd = False

prompt = "a cowboy riding a horse"
prompt = "a cowboy riding a horse with a city in the background"

def to_im(im):
    return Image.fromarray((im * 255).clip(0, 255).astype("uint8"))

# for motion_field_strength_x in [0, 5, 10, 15, 20, 25]:
# for motion_field_strength_x in [40, 60, 80]:
# for motion_field_strength_x in [10]:
for motion_field_strength_x in [-15]:
    # for t in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
    # for t0 in [3, 2, 1]:
    for t1 in [
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]:
        # for dt in [0, 1, 2]:
        # for dt in [0]:

        sd_image, _, muse_image, muse_latents = pipe(
            prompt=prompt, 
            # video_length=12, 
            # video_length=8, 
            video_length=2,
            # generator=torch.Generator('cuda').manual_seed(0),
            motion_field_strength_y=0, 
            # motion_field_strength_x=40,
            # motion_field_strength_x=0,
            motion_field_strength_x=motion_field_strength_x,
            # t0=47,
            # t1=47,
            # muse_t0=t0,
            # muse_t1=t0+dt,
            muse_t1=t1,
            # muse_generator=torch.Generator('cuda').manual_seed(0),
            muse_generator=torch.Generator('cuda').manual_seed(1),
            return_dict=False,
        )

        orig_muse_image = muse_image
        orig_muse_latents = muse_latents

        # for seed in range(4, 10):
        for seed in [5]:

            all_gif_frames = [orig_muse_image[0]]
            muse_latents = orig_muse_latents

            # for _ in range(12):
            # for _ in range(20):
            for _ in range(17):
            # for _ in range(1):
                muse_latents = muse_latents[1].unsqueeze(0)
                muse_latents = pipe.muse_pipe.scheduler.add_noise(muse_latents, 1, generator=torch.Generator('cuda').manual_seed(0))

                muse_pipe.scheduler.frames = True
                muse_pipe.scheduler.starting_mask_ratio = float((muse_latents == 8255).sum()) / float(muse_latents.numel())

                sd_image, _, muse_image, muse_latents = pipe(
                    prompt=prompt, 
                    # video_length=12, 
                    # video_length=8, 
                    video_length=2,
                    # generator=torch.Generator('cuda').manual_seed(0),
                    motion_field_strength_y=0, 
                    # motion_field_strength_x=40,
                    # motion_field_strength_x=0,
                    motion_field_strength_x=motion_field_strength_x,
                    # t0=47,
                    # t1=47,
                    # muse_t0=t0,
                    # muse_t1=t0+dt,
                    muse_t1=[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                    # muse_generator=torch.Generator('cuda').manual_seed(0),
                    # muse_generator=torch.Generator('cuda').manual_seed(0),
                    # muse_generator=torch.Generator('cuda').manual_seed(1),
                    muse_generator=torch.Generator('cuda').manual_seed(seed),
                    return_dict=False,
                    muse_latents=muse_latents,
                )

                all_gif_frames.append(muse_image[0])

            # assert (sd_image[0] == sd_image[1]).all()
            # assert (muse_image[0] == muse_image[1]).all()

            muse_image = [to_im(x) for x in all_gif_frames]
            # muse_image[0].save(f"muse_motion_field_x_{motion_field_strength_x}_t0_{t0}_dt_{dt}.gif", save_all=True, append_images=muse_image[1:], duration=150, loop=0)
            muse_image[0].save(f"muse_text_to_video_incremental_2_motion_field_x_{motion_field_strength_x}_t1_{t1}_seed_{seed}.gif", save_all=True, append_images=muse_image[1:], duration=150, loop=0)
