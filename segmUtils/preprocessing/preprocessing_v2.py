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

import pandas
import zarr

from segmfriends.io.zarr import append_arrays_to_zarr

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
                           max_nb_crops_per_image=None, process_channels=True, rename_unique=True,
                           BF_ch_filter="_c0", DAPI_ch_filter="_c1", general_filter=None):
    """

    :param input_dir_path:
    :param out_dir_path:
    :param crop_size:
    :param projectdir_depth:
    :param delete_out_dir:
    :param starting_index:
    :param max_nb_images: only collect a max amount of images and then stop
    :param process_channels: Whether to expect images with BF_ch_filter and DAPI_ch_filter patterns
    :param rename_unique:
    :param max_nb_crops_per_image: only collect a given amount of crops per image
    :param BF_ch_filter:
    :param DAPI_ch_filter:
    :param general_filter: a string that should be present in all processed image files, used to mask some of the
                        images (useful expecially if process_channels is False, but we want to skip certain images)
    :return:
    """

    check_dir_and_create(out_dir_path)
    if delete_out_dir:
        shutil.rmtree(out_dir_path)
        check_dir_and_create(out_dir_path)

    cellpose_out = os.path.join(out_dir_path, "cellpose")
    livecell_out = os.path.join(out_dir_path, "LIVECell")
    check_dir_and_create(cellpose_out)
    check_dir_and_create(livecell_out)

    def write_image_cellpose(img, crop_slice, out_dir, out_name):
        # Write Cell-Pose image:
        img = img[crop_slice]
        cv2.imwrite(os.path.join(out_dir, out_name), img)

        z_cellpose_path = os.path.join(out_dir_path, "cellpose.zarr")
        append_arrays_to_zarr(z_cellpose_path, add_array_dimensions=True,
                              BF=img[...,1], DAPI=img[...,2])



    def write_image_livecell(img, crop_slice, out_dir, out_name):
        # Write LIVECell image:
        pass
        # cv2.imwrite(os.path.join(out_dir, out_name), preprocess_LIVEcell.preprocess(img[crop_slice],
        #                                                                             magnification_downsample_factor=1))


    def get_image_paths():
        idx_images = starting_index

        # Initialize lists for creating pandas dataframe:
        all_image_paths = []

        for root, dirs, files in os.walk(input_dir_path):
            for filename in files:
                file_basename, file_extension = os.path.splitext(filename)
                if not process_channels or file_basename.endswith(BF_ch_filter):
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

                        # Save original image path:
                        new_image_paths = []
                        input_image_paths = [input_dir_path, os.path.relpath(image_path, input_dir_path), None]

                        # Check if there is a DAPI channel file:
                        if process_channels:
                            dapi_image = os.path.join(root, filename.replace(BF_ch_filter, DAPI_ch_filter))
                            if os.path.exists(dapi_image):
                                img_dapi = read_uint8_img(dapi_image)
                                assert img_dapi.shape == shape

                                # Rescale to bring up signal of DAPI channel:
                                max_dapi = img_dapi.max()
                                img_dapi = (img_dapi.astype('float32') * 255. / max_dapi).astype('uint8')

                                cellpose_image[...,2] = img_dapi[...,0]

                                # Save DAPI image path:
                                input_image_paths[2] = os.path.relpath(dapi_image, input_dir_path)


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
                                out_filename = "{}_{}_{}_{}.png".format(
                                    idx_images,
                                    prj_name,
                                    filename.replace(file_extension, ""),
                                    slice_idx
                                )

                                # Check if we should stop here:
                                if max_nb_images is not None:
                                    if idx_images - starting_index >= max_nb_images:
                                        all_image_paths += new_image_paths
                                        return all_image_paths, idx_images
                                if max_nb_crops_per_image is not None:
                                    if slice_idx >= max_nb_crops_per_image:
                                        break

                                # Write Cell-Pose image:
                                write_image_cellpose(cellpose_image, crop_slc, cellpose_out, out_filename)

                                # Write LIVECell image:
                                write_image_livecell(img, crop_slc, livecell_out, out_filename)
                                idx_images += 1

                                # Save out image paths:
                                new_image_paths.append(
                                    input_image_paths + [os.path.join(cellpose_out, out_filename), os.path.join(livecell_out, out_filename)]
                                )

                        else:
                            assert crop_size is None, "When applying crops, image names must be unique"
                            out_name = "{}.png".format(file_basename)
                            write_image_cellpose(cellpose_image, slice(None), cellpose_out, out_name)
                            write_image_livecell(img, slice(None), livecell_out, out_name)
                            idx_images += 1

                            # Save out image paths:
                            new_image_paths.append(
                                input_image_paths + [os.path.join(cellpose_out, out_name),
                                                     os.path.join(livecell_out, out_name)]
                            )
                        all_image_paths += new_image_paths
        return all_image_paths, idx_images
    all_image_paths, idx_images = get_image_paths()
    print("{} images saved in {}".format(idx_images, out_dir_path))

    df = pandas.DataFrame(data=all_image_paths, columns=["Input-dir", "Input-BF", "Input-DAPI", "Out-Cellpose", "Out-LIVECell"])
    df.to_csv(os.path.join(out_dir_path, "image_paths.csv"), index=False)
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

    # # ----------------------
    # # Get new glioblastoma images from Alex:
    # # ----------------------
    # out_dir = os.path.join("/scratch/bailoni/datasets/alex/glioblastoma/preprocessed")
    # process_images_in_path("/scratch/bailoni/datasets/alex/glioblastoma/images", out_dir, process_channels=False,
    #                        rename_unique=False)

    # ----------------------
    # Get new glioblastoma images from Alex v2:
    # ----------------------
    # process_images_in_path("/scratch/bailoni/datasets/alex/glioblastoma-v2/data",
    #                         "/scratch/bailoni/datasets/alex/glioblastoma-v2/preprocessed", process_channels=False,
    #                        rename_unique=True)
    # process_images_in_path("/scratch/bailoni/datasets/martijn/examplesMacrophages/data",
    #                         "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_ch2", process_channels=True,
    #                        rename_unique=True, max_nb_crops_per_image=1, BF_ch_filter="_c2", delete_out_dir=True)
    # process_images_in_path("/scratch/bailoni/datasets/martijn/examplesMacrophages/data",
    #                         "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BF", process_channels=True,
    #                        rename_unique=True, max_nb_crops_per_image=1, delete_out_dir=True)
    process_images_in_path("/scratch/bailoni/datasets/martijn/examplesMacrophages/data",
                            "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2", process_channels=True,
                           rename_unique=True, max_nb_crops_per_image=1, BF_ch_filter="_c0", DAPI_ch_filter="_c2", delete_out_dir=True)

    print("Done")





