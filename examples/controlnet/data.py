import cv2
import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import default_collate
from torchvision import transforms
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def control_transform(image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


def make_wds_canny_controlnet_dataset(
    train_shards_path_or_url: str,
    per_gpu_batch_size: int,
    resolution: int = 256,
    shuffle_buffer_size: int = 1000,
):
    def get_orig_size(json):
        return (int(json.get("original_width", 0.0)), int(json.get("original_height", 0.0)))

    pipeline = [
        wds.ResampledShards(train_shards_path_or_url),
        tarfile_to_samples_nothrow,
        wds.shuffle(shuffle_buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.rename(
            image="jpg;png;jpeg;webp",
            control_image="jpg;png;jpeg;webp",
            text="text;txt;caption",
            orig_size="json",
            handler=wds.warn_and_continue,
        ),
        wds.map(filter_keys({"image", "control_image", "text", "orig_size"})),
        wds.map_dict(
            image=transforms.Compose(
                [
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
            control_image=transforms.Compose(
                [
                    control_transform,
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                ]
            ),
            orig_size=get_orig_size,
        ),
        wds.to_tuple("image", "control_image", "text", "orig_size"),
        wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
    ]

    dataset = wds.DataPipeline(*pipeline)

    return dataset
