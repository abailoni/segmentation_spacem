import numpy as np

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random

from segmfriends.utils.various import check_dir_and_create
try:
    from ..LIVECellutils.preprocessing import preprocess as preprocess_LIVEcell
except ImportError:
    from segmUtils.LIVECellutils.preprocessing import preprocess as preprocess_LIVEcell

def process_images_in_path(input_dir_path, out_dir_path, crop_size=None):
    idx_images = 0

    cellpose_out = os.path.join(out_dir_path, "cellpose")
    livecell_out = os.path.join(out_dir_path, "LIVECell")
    check_dir_and_create(cellpose_out)
    check_dir_and_create(livecell_out)

    for root, dirs, files in os.walk(input_dir_path):
        for filename in files:
            if filename.endswith("_c0.tif"):
                # Ignore post-maldi images for the moment:
                if "cropped_post_maldi_channels" not in os.path.split(root)[1]:
                    image_path = os.path.join(root, filename)
                    # By default, BGR is read:
                    img = cv2.imread(image_path)
                    shape = img.shape # Example: (x, y, 3)

                    # Cell-Pose training setup:
                    # - Green channel: image
                    # - Red channel: DAPI
                    cellpose_image = img.copy()
                    cellpose_image[...,0] = 0
                    cellpose_image[...,2] = 0

                    # Check if there is a DAPI channel file:
                    dapi_image = os.path.join(root, filename.replace("_c0", "_c1"))
                    if os.path.exists(dapi_image):
                        img_dapi = cv2.imread(dapi_image)
                        assert img_dapi.shape == shape

                        # Rescale to bring up signal of DAPI channel:
                        max_dapi = img_dapi.max()
                        img_dapi = (img_dapi.astype('float32') * 255. / max_dapi).astype('uint8')

                        cellpose_image[...,2] = img_dapi[...,0]

                    # Compose unique new name for output image:
                    prj_name = os.path.split(root)[1]
                    if "cropped_pre_maldi_channels" in os.path.split(root)[1]:
                        # Deduce name based on well number and project folder (assume usual SpaceM directory structure):
                        prj_name = "{}_{}".format(root.split("/")[-4], root.split("/")[-5])

                    # If no crop size is given, take the full image as it is:
                    crop_size = shape[:2] if crop_size is None else crop_size

                    nb_channels = shape[2]
                    full_crop_size = tuple(crop_size) + (nb_channels,)
                    window_slices = slidingwindowslices(shape, full_crop_size, strides=full_crop_size,
                                        shuffle=False, add_overhanging=True)
                    for slice_idx, slice in enumerate(window_slices):
                        out_filename = "{}_{}_{}_{}_img.png".format(
                            idx_images,
                            prj_name,
                            filename.replace(".tif", ""),
                            slice_idx
                        )

                        # Write Cell-Pose image:
                        cropped_image = cellpose_image[slice]
                        cv2.imwrite(os.path.join(cellpose_out, out_filename), cropped_image)

                        # Write LIVECell image:
                        cv2.imwrite(os.path.join(livecell_out, out_filename), preprocess_LIVEcell(img[slice]))
                        idx_images += 1
    print("{} images saved in {}".format(idx_images, out_dir_path))




if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"
    input_dir = os.path.join(scratch_dir, "datasets/alyona/20210823_AB_DKFZCocultur_analysis")
    out_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images")
    process_images_in_path(input_dir, out_dir, crop_size=(520, 704))
