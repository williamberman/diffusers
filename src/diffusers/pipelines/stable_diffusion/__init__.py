from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_version,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


if is_transformers_available() and is_torch_available():
    pass

try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    pass


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    pass


try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_version(">=", "0.0.12")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    pass

if is_transformers_available() and is_onnx_available():
    pass

if is_transformers_available() and is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionPipelineOutput(BaseOutput):
        """
        Output class for Stable Diffusion pipelines.

        Args:
            images (`np.ndarray`)
                Array of shape `(batch_size, height, width, num_channels)` with images from the diffusion pipeline.
            nsfw_content_detected (`List[bool]`)
                List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
                (nsfw) content.
        """

        images: np.ndarray
        nsfw_content_detected: List[bool]
