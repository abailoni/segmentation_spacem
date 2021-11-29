import cv2
import numpy as np


def convert_to_cellpose_style(input_image, method="subtract"):
    # internal variables
    #   median_radius_raw = used in the background illumination pattern estimation.
    #       this radius should be larger than the radius of a single cell
    median_radius_raw = 75

    magnification_downsample_factor = 1.

    # large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw * magnification_downsample_factor)
    if median_radius % 2 == 0:
        median_radius = median_radius + 1

    output_image = input_image.copy()

    # estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    if len(background.shape) < len(output_image.shape):
        # Add back channel dimension:
        background = background[..., None]

    # Take difference, abs value and move back again to [0, 255] interval:
    if method == "subtract":
        output_image = background.astype('float') - output_image.astype('float')
        output_image = np.abs(output_image)
    elif method == "multiply":
        # Add small epsilon to avoid nan in the division:
        background = background.astype("float32") + 0.01
        output_image = output_image.astype('float') / background.astype('float')
        # Move 1. to zero and take absolute max:
        output_image -= 1.
        output_image = np.abs(output_image)
    elif method == "shift":
        # Move 128. to zero and take absolute max:
        output_image -= 128.
        output_image = np.abs(output_image)
    else:
        raise NotImplementedError
    output_image = output_image - output_image.min()
    output_image = output_image / output_image.max() * 255.

    output_image = output_image.astype('uint8')

    return output_image
