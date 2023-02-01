from ..utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
)


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_pt_objects import *  # noqa F403
else:
    pass

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    pass

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    pass

try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_onnx_objects import *  # noqa F403
else:
    pass

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    pass

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    pass

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_objects import *  # noqa F403
else:
    pass


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    pass
