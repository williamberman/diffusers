import torch
from diffusers.pipelines.open_muse.pipeline_open_muse_text_to_video_zero import OpenMuseTextToVideoZeroPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

model_id = "../muse-512-finetuned-convert"
pipe = OpenMuseTextToVideoZeroPipeline.from_pretrained(model_id).to("cuda")
pipe.transformer.set_attn_processor(AttnProcessor())


embedding_layer = pipe.vqvae.quantize.embedding
n_new_embeddings = pipe.scheduler.config.mask_token_id - embedding_layer.num_embeddings + 1
new_embeddings = torch.randn(n_new_embeddings, embedding_layer.embedding_dim, device='cuda')
extended_weight = torch.cat([embedding_layer.weight, new_embeddings], 0)
embedding_layer.num_embeddings += n_new_embeddings
embedding_layer.weight = torch.nn.Parameter(extended_weight)

prompt = "a cowboy riding a horse with a city in the background"

def to_im(im):
    return Image.fromarray((im * 255).clip(0, 255).astype("uint8"))

for motion_field_strength_x in [-15]:
    for t1 in [
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]:
        muse_image, muse_latents = pipe(
            prompt=prompt, 
            video_length=2,
            motion_field_strength_y=0, 
            motion_field_strength_x=motion_field_strength_x,
            muse_t1=t1,
            muse_generator=torch.Generator('cuda').manual_seed(1),
            return_dict=False,
        )

        orig_muse_image = muse_image
        orig_muse_latents = muse_latents

        for seed in [5]:

            all_gif_frames = [orig_muse_image[0]]
            muse_latents = orig_muse_latents

            for _ in range(17):
                muse_latents = muse_latents[1].unsqueeze(0)
                muse_latents = pipe.scheduler.add_noise(muse_latents, 1, generator=torch.Generator('cuda').manual_seed(0))

                pipe.scheduler.frames = True
                pipe.scheduler.starting_mask_ratio = float((muse_latents == 8255).sum()) / float(muse_latents.numel())

                muse_image, muse_latents = pipe(
                    prompt=prompt, 
                    video_length=2,
                    motion_field_strength_y=0, 
                    motion_field_strength_x=motion_field_strength_x,
                    muse_t1=[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                    muse_generator=torch.Generator('cuda').manual_seed(seed),
                    return_dict=False,
                    muse_latents=muse_latents,
                )

                all_gif_frames.append(muse_image[0])

            muse_image = [to_im(x) for x in all_gif_frames]
            muse_image[0].save(f"muse_text_to_video_incremental_2_motion_field_x_{motion_field_strength_x}_t1_{t1}_seed_{seed}.gif", save_all=True, append_images=muse_image[1:], duration=150, loop=0)
