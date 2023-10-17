from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        OpenMuseImg2ImgPipeline,
        OpenMuseInpaintPipeline,
        OpenMusePipeline,
    )

    _dummy_objects.update(
        {
            "OpenMusePipeline": OpenMusePipeline,
            "OpenMuseImg2ImgPipeline": OpenMuseImg2ImgPipeline,
            "OpenMuseInpaintPipeline": OpenMuseInpaintPipeline,
        }
    )
else:
    _import_structure["pipeline_open_muse"] = ["OpenMusePipeline"]
    _import_structure["pipeline_open_muse_img2img"] = ["OpenMuseImg2ImgPipeline"]
    _import_structure["pipeline_open_muse_inpaint"] = ["OpenMuseInpaintPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            OpenMusePipeline,
        )
    else:
        from .pipeline_open_muse import OpenMusePipeline
        from .pipeline_open_muse_img2img import OpenMuseImg2ImgPipeline
        from .pipeline_open_muse_inpaint import OpenMuseInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
