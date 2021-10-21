import numpy as np

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import shutil

from segmfriends.utils.various import check_dir_and_create
try:
    from ..LIVECellutils import preprocessing as preprocess_LIVEcell
except ImportError:
    from segmUtils.LIVECellutils import preprocessing as preprocess_LIVEcell

def read_uint8_img(img_path):
    img = cv2.imread(img_path)
    # Sometimes some images are loaded in float and cannot be automatically converted to uint8:
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = img - img.min()
        img = (img / img.max() * 255.).astype('uint8')

    if len(img.shape) == 2:
        # Add channel dimension:
        img = np.stack([img for _ in range(3)])
        img = np.rollaxis(img, axis=0, start=3)
    assert len(img.shape) == 3 and img.shape[2] == 3, img.shape
    return img



def process_images_in_path(input_dir_path, out_dir_path, crop_size=None, projectdir_depth=None,
                           delete_out_dir=False, starting_index=0, max_nb_images=None,
                           filter_c0_images=True, rename_unique=True):
    idx_images = starting_index

    check_dir_and_create(out_dir_path)
    if delete_out_dir:
        shutil.rmtree(out_dir_path)
        check_dir_and_create(out_dir_path)

    cellpose_out = os.path.join(out_dir_path, "cellpose")
    livecell_out = os.path.join(out_dir_path, "LIVECell")
    check_dir_and_create(cellpose_out)
    check_dir_and_create(livecell_out)

    def write_image_cellpose(img, slice, out_dir, out_name):
        # Write Cell-Pose image:
        cropped_image = img[slice]
        cv2.imwrite(os.path.join(out_dir, out_name), cropped_image)

    def write_image_livecell(img, slice, out_dir, out_name):
        # Write LIVECell image:
        cv2.imwrite(os.path.join(out_dir, out_name), preprocess_LIVEcell.preprocess(img[slice],
                                                                                             magnification_downsample_factor=1))

    for root, dirs, files in os.walk(input_dir_path):
        for filename in files:
            file_basename, file_extension = os.path.splitext(filename)
            if not filter_c0_images or file_basename.endswith("_c0"):
                # Ignore post-maldi images for the moment:
                if "cropped_post_maldi_channels" not in os.path.split(root)[1]:
                    image_path = os.path.join(root, filename)
                    # By default, BGR is read:
                    img = read_uint8_img(image_path)
                    shape = img.shape # Example: (x, y, 3)

                    # Cell-Pose training setup:
                    # - Green channel: image
                    # - Red channel: DAPI
                    cellpose_image = img.copy()
                    cellpose_image[...,0] = 0
                    cellpose_image[...,2] = 0

                    # Check if there is a DAPI channel file:
                    if filter_c0_images:
                        dapi_image = os.path.join(root, filename.replace("_c0", "_c1"))
                        if os.path.exists(dapi_image):
                            img_dapi = read_uint8_img(dapi_image)
                            assert img_dapi.shape == shape

                            # Rescale to bring up signal of DAPI channel:
                            max_dapi = img_dapi.max()
                            img_dapi = (img_dapi.astype('float32') * 255. / max_dapi).astype('uint8')

                            cellpose_image[...,2] = img_dapi[...,0]

                    # Compose unique new name for output image:
                    if "cropped_pre_maldi_channels" in os.path.split(root)[1] and projectdir_depth is None:
                        # Deduce name based on well number and project folder (assume usual SpaceM directory structure):
                        prj_name = "{}_{}".format(root.split("/")[-4], root.split("/")[-5])
                    elif projectdir_depth is None:
                        prj_name = os.path.split(root)[1]
                    else:
                        prj_name = root.split("/")[projectdir_depth]

                    if rename_unique:
                        # If no crop size is given, take the full image as it is:
                        applied_crop_size = shape[:2] if crop_size is None else crop_size

                        nb_channels = shape[2]
                        applied_crop_size = tuple(applied_crop_size) + (nb_channels,)
                        window_slices = slidingwindowslices(shape, applied_crop_size, strides=applied_crop_size,
                                            shuffle=False, add_overhanging=True)
                        for slice_idx, crop_slc in enumerate(window_slices):
                            out_filename = "{}_{}_{}_{}_img.png".format(
                                idx_images,
                                prj_name,
                                filename.replace(file_extension, ""),
                                slice_idx
                            )

                            # Check if we should stop here:
                            if max_nb_images is not None:
                                if idx_images - starting_index >= max_nb_images:
                                    print("{} images saved in {}".format(idx_images, out_dir_path))
                                    return idx_images

                            # Write Cell-Pose image:
                            write_image_cellpose(cellpose_image, crop_slc, cellpose_out, out_filename)

                            # Write LIVECell image:
                            write_image_livecell(img, crop_slc, livecell_out, out_filename)
                            idx_images += 1
                    else:
                        assert crop_size is None, "When applying crops, image names must be unique"
                        write_image_cellpose(cellpose_image, slice(None), cellpose_out, "{}.png".format(file_basename))
                        write_image_livecell(img, slice(None), livecell_out, "{}.png".format(file_basename))
                        idx_images += 1
    print("{} images saved in {}".format(idx_images, out_dir_path))
    return idx_images




if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"

    # ----------------------
    # Alyona images:
    # ----------------------
    # out_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images")
    # nb_images = process_images_in_path(os.path.join(scratch_dir, "datasets/alyona/20210823_AB_DKFZCocultur_analysis"),
    #                                    out_dir, delete_out_dir=True)

    # ----------------------
    # Collect some crops from given images:
    # ----------------------
    # out_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images_small")
    # nb_images = process_images_in_path("/scratch/bailoni/datasets/alex/210920_prostate-v1_cellpose-training", out_dir,
    #                                    delete_out_dir=True)
    # nb_images = process_images_in_path(os.path.join(scratch_dir, "datasets/alyona/20210823_AB_DKFZCocultur_analysis"),
    #                                    out_dir, crop_size=(800, 500), starting_index=nb_images,
    #                                    max_nb_images=4)
    # nb_images = process_images_in_path("/scratch/bailoni/datasets/martijn/CellSegmentationExamples", out_dir, crop_size=(1000, 1000),
    #                        projectdir_depth=-2, starting_index=nb_images)

    # # ----------------------
    # # Get all labelled images from Alex:
    # # ----------------------
    # out_dir = os.path.join(scratch_dir, "projects/spacem_segm/alex_labeled")
    # process_images_in_path("/scratch/bailoni/datasets/alex/210920_prostate-v1_cellpose-training", out_dir)

    # ----------------------
    # Get new glioblastoma images from Alex:
    # ----------------------
    out_dir = os.path.join("/scratch/bailoni/datasets/alex/glioblastoma/preprocessed")
    process_images_in_path("/scratch/bailoni/datasets/alex/glioblastoma/images", out_dir, filter_c0_images=False,
                           rename_unique=False)



