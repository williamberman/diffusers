from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin

from .attention import CrossAttention


class VQDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    Transformer from VQ Diffusion. Takes classes of latent pixels for a given timestep in the diffusion process and
    outputs a probability distribution of the completely unnoised latent pixels. Note that there is no prediction for
    the masked latent pixel class as the unnoised image cannot be masked.

    For more details, see the original paper: https://arxiv.org/abs/2111.14822

    Args:
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`): The number of layers of Transformer blocks to use.
        context_dim (:obj:`int`): The number of context dimensions to use.
        num_embed (:obj:`int`):
            The number of classes of the vector embeddings of the latent pixels. Includes the class for the masked
            latent pixel.
        height (:obj: `int`):
            The height of the latent images. Note that this is fixed at training time as it is used for learning a
            number of position embeddings. See `ImageEmbeddings`.
        width (:obj: `int`):
            The width of the latent images. Note that this is fixed at training time as it is used for learning a
            number of position embeddings. See `ImageEmbeddings`.
        diffusion_steps (:obj: `int`):
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than `diffusion_steps`.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
    """

    @register_to_config
    def __init__(
        self,
        n_heads: int,
        d_head: int,
        depth: int,
        context_dim: int,
        num_embed: int,
        height: int,
        width: int,
        diffusion_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head
        self.inner_dim = n_heads * d_head
        self.num_embed = num_embed
        self.height = height
        self.width = width
        self.num_latent_pixels = self.height * self.width

        self.latent_image_embedding = VQDiffusionImageEmbeddings(
            num_embed=self.num_embed, embed_dim=self.inner_dim, height=self.height, width=self.width
        )

        self.transformer_blocks = nn.ModuleList(
            [
                VQDiffusionTransformerBlock(
                    self.inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    diffusion_steps=diffusion_steps,
                )
                for d in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim)

        self.out = nn.Linear(self.inner_dim, self.num_embed - 1)

    def forward(self, latent_images, cond_emb, t):
        """
        Args:
            latent_images (:obj: `torch.LongTensor` of shape `(batch size, num latent pixels)`):
                Input latents to be denoised.
            cond_emb (:obj: `torch.LongTensor` of shape `(batch size, context dim)`):
                Conditional embeddings for cross attention layer.
            t (:obj: `torch.long`):
                Denoising timestep

        Returns:
            `torch.FloatTensor` of shape `(batch size, num embed - 1, num latent pixels)`:
                Probability distributions for the unnoised latent pixels. Note that it does not output a prediction for
                the masked class.
        """
        embedded_latent_images = self.latent_image_embedding(latent_images)
        hidden_states = embedded_latent_images

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, cond_emb, t)

        logits = self.out(self.norm_out(hidden_states))
        # (batch, self.num_embed - 1, self.num_latent_pixels)
        logits = logits.permute(0, 2, 1)

        log_p_x_0 = F.log_softmax(logits.double(), dim=1).float()

        return log_p_x_0


class VQDiffusionImageEmbeddings(nn.Module):
    """
    Converts latent image classes into vector embeddings for the transformer. Sums the vector embeddings with
    positional embeddings for the height and width of the latent space.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        emb = self.emb(index)

        height_emb = self.height_emb(torch.arange(self.height, device=index.device).view(1, self.height))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(torch.arange(self.width, device=index.device).view(1, self.width))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class VQDiffusionTransformerBlock(nn.Module):
    """
    A basic transformer block modified to take the timestep of the diffusion process as additional input.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        diffusion_steps: int,
        dropout=0.0,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=True
        )  # is a self-attention
        self.ff = VQDiffusionFeedForward(dim=dim, dropout=dropout)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=True
        )
        self.norm1 = AdaLayerNorm(dim, diffusion_steps)
        self.norm2 = AdaLayerNorm(dim, diffusion_steps)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden_states, context, timestep):
        hidden_states = self.attn1(self.norm1(hidden_states, timestep)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states, timestep), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class VQDiffusionFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.Sequential(nn.Linear(dim, inner_dim), GELU2(), nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
