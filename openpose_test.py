from PIL import Image
from controlnet_aux import OpenposeDetector
import numpy as np
from controlnet_aux.util import HWC3, resize_image

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

def any_poses_detected(input_image):
    input_image = np.array(input_image, dtype=np.uint8)

    input_image = HWC3(input_image)
    input_image = resize_image(input_image, 512)
    poses = openpose.detect_poses(input_image)

    return len(poses) > 0

for i in range(4):
    image = Image.open(f'./t2i_openpose_validation/{i}_input.png')

    if any_poses_detected(image):
        print(f'poses detected {i}')
        image = openpose(image)
        image.save(f'./t2i_openpose_validation/{i}.png')
    else:
        print(f'no poses detected {i}')