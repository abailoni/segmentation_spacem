import numpy as np
from speedrun import locate

from segmUtils.utils.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import shutil

from segmUtils.preprocessing.process_images import convert_to_cellpose_style
from segmfriends.utils.various import check_dir_and_create
try:
    from ..LIVECellutils import preprocessing as preprocess_LIVEcell
except ImportError:
    from segmUtils.LIVECellutils import preprocessing as preprocess_LIVEcell

import pandas
import zarr
import imageio
from PIL import Image, ImageEnhance

import segmfriends.io.zarr as zarr_utils
import segmfriends.utils.various as var_utils

def read_segmentation_from_file(img_path):
    assert os.path.isfile(img_path), "Image {} not found".format(img_path)
    img = imageio.imread(img_path)
    return img

def read_uint8_img(img_path, add_all_channels_if_needed=True):
    # TODO: rename and move to io module together with function exporting segmentation file
    assert os.path.isfile(img_path), "Image {} not found".format(img_path)

    extension = os.path.splitext(img_path)[1]
    if extension == ".tif" or extension == ".tiff":
        # img = cv2.imread(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # print(img.dtype, img.min(), img.max())
        # Sometimes some images are loaded in float and cannot be automatically converted to uint8:
        # FIXME: check type and then convert to uint8 (or uint16??)
        if img.dtype == 'uint16':
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        assert img.dtype == 'uint8'
            # # img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # print(img.dtype, img.min(), img.max())
            # img = img - img.min()
            # img = (img / img.max() * 255.).astype('uint8')
    elif extension == ".png":
        img = imageio.imread(img_path)
    else:
        raise ValueError("Extension {} not supported".format(extension))
    if len(img.shape) == 2 and add_all_channels_if_needed:
        # Add channel dimension:
        img = np.stack([img for _ in range(3)])
        img = np.rollaxis(img, axis=0, start=3)
    # assert len(img.shape) == 3 and img.shape[2] == 3, img.shape

    return img


def apply_preprocessing_to_image(image, ch_name, preprocessing_specs):
    if ch_name in preprocessing_specs:
        all_prep_funcs = preprocessing_specs[ch_name]
        all_prep_funcs = all_prep_funcs if isinstance(all_prep_funcs, list) else [all_prep_funcs]
        for prep_fct_specs in all_prep_funcs:
            assert isinstance(prep_fct_specs, dict)
            prep_kwargs = prep_fct_specs["function_kwargs"]
            print("Applying {} preproc function...".format(prep_fct_specs["function_name"]))
            preprocessing_function = locate(prep_fct_specs["function_name"], [])
            image = preprocessing_function(image, **prep_kwargs)
    return image

def convert_images_to_zarr_dataset(input_dir_path, out_zarr_path=None, crop_size=None, projectdir_depth=None,
                                   starting_index=0, max_nb_images=None,
                                   ensure_all_channel_existance=True,
                                   rename_unique=True, max_nb_crops_per_image=None,
                                   general_filter=None,
                                   filename_filter_exclude=None,
                                   filename_filter_include=None,
                                   precrop=None,
                                   dataset_name=None,
                                   extension=None,
                                   delete_previous_zarr=True,
                                   save_to_zarr=True,
                                   verbose=False,
                                   preprocessing=None,
                                   **channels_filters):
    # TODO: give a better name (considering options to only get paths in a folder)
    """
    :param channels_filters: Dictionary of channel names and associated ending-filename-filters. For example:
                {BF: "_ch0", DAPI: "_ch1"}. If all images should be processed as single channel, a None filter can be
                passed, e.g. {main_channel_name: None}.
                If a channel should be saved as zeros (to keep consistency with other datasets), pass "$ZERO" as a
                filter, e.g. {BF: "_c0", DAPI: "$ZERO"}. TODO: not compatible with None main channel at the moment.
    :param dataset_name: used for when multiple datasets are inserted in the same zarr file
    :param delete_previous_zarr: by default, set to True delete previous to avoid inconsistencies. Can be set to False,
            but use with care.
    """
    preprocessing = {} if preprocessing is None else preprocessing
    assert isinstance(preprocessing, dict)

    if filename_filter_exclude is not None:
        if not isinstance(filename_filter_exclude, (list, tuple)):
            filename_filter_exclude = [filename_filter_exclude]
    if filename_filter_include is not None:
        if not isinstance(filename_filter_include, (list, tuple)):
            filename_filter_include = [filename_filter_include]

    if save_to_zarr:
        assert isinstance(out_zarr_path, str)

        # TODO: remove valid mask from zarr file (now stored in csv)
        # Delete previous images in the zarr group:
        if os.path.exists(out_zarr_path) and delete_previous_zarr:
            shutil.rmtree(out_zarr_path)

    assert len(channels_filters), "No channel names passed!"

    if any([channels_filters[ch_name] is None for ch_name in channels_filters]):
        assert len(channels_filters) == 1, "Onyl one channel can be given without filename-filter!"

    main_ch_name = [ch_name for ch_name in channels_filters][0]
    main_ch_filter = channels_filters[main_ch_name]

    dataset_name = "" if dataset_name is None else dataset_name

    def get_image_paths():
        def read_image(img_path):
            img = read_uint8_img(img_path)[..., 0]
            if precrop is not None:
                img = img[var_utils.parse_data_slice(precrop)]
            return img

        idx_images = starting_index

        # Initialize lists for creating pandas dataframe:
        path_headers = None
        all_image_paths = []

        for root, dirs, files in os.walk(input_dir_path):
            for filename in files:
                file_basename, file_extension = os.path.splitext(filename)
                if file_basename.endswith(main_ch_filter) or main_ch_filter is None:
                    # Check for general filter:
                    if general_filter is not None:
                        if general_filter not in file_basename:
                            continue

                    full_path = os.path.join(root, filename)
                    excluded = False
                    # Check if we should skip this file:
                    if filename_filter_include is not None:
                        for filt in filename_filter_include:
                            if filt not in full_path:
                                excluded = True
                                break

                    if filename_filter_exclude is not None and not excluded:
                        for filt in filename_filter_exclude:
                            if filt in full_path:
                                excluded = True
                                break

                    if excluded:
                        continue

                    # Check for correct extension of raw file:
                    if extension is not None:
                        if file_extension != extension:
                            continue

                    # Ignore post-maldi images for the moment:
                    if "cropped_post_maldi_channels" in os.path.split(root)[1]:
                        continue
                    main_ch_path = os.path.join(root, filename)
                    # By default, BGR is read, so remove channel dimension:
                    main_ch_img = read_image(main_ch_path)
                    # print(main_ch_name, main_ch_img.max(), main_ch_img.min(), main_ch_img.mean())
                    main_ch_img = apply_preprocessing_to_image(main_ch_img, main_ch_name, preprocessing)

                    shape = main_ch_img.shape

                    # Save main channel in zarr file:
                    zarr_kwargs = {main_ch_name: main_ch_img}

                    # Save original image path:
                    new_image_paths = []
                    new_path_headers = ["Input dir", "Dataset name", main_ch_name]
                    input_image_paths = [input_dir_path, dataset_name, os.path.relpath(main_ch_path, input_dir_path)]

                    for ch_name, ch_filter in channels_filters.items():
                        if ch_name != main_ch_name:
                            if ch_filter == "$ZERO":
                                pass
                                new_ch_img = np.zeros_like(main_ch_img)
                                new_image_path = ""
                            else:
                                # Load extra channel and save it to zarr dataset:
                                image_path = os.path.join(root, filename.replace(main_ch_filter, ch_filter))
                                if not os.path.exists(image_path):
                                    if ensure_all_channel_existance:
                                        raise ValueError("Channel {} not found for image {} in {}".format(ch_name, filename, root))
                                    else:
                                        continue
                                # print(ch_name)
                                new_ch_img = read_image(image_path)
                                if new_ch_img.shape != shape:
                                    print("Warning! Image channels have different dimensions!",
                                          "{}: {}, Main channel: {}. Image path: {}".format(
                                              ch_name, new_ch_img.shape, shape, image_path)
                                          )
                                # assert new_ch_img.shape == shape, "{}: {}, Main channel: {}. Image path: {}".format(
                                #     ch_name, new_ch_img.shape, shape,image_path)
                                print(ch_name, new_ch_img.max(), new_ch_img.min(), new_ch_img.mean())
                                new_ch_img = apply_preprocessing_to_image(new_ch_img, ch_name, preprocessing)
                                new_image_path = os.path.relpath(image_path, input_dir_path)
                            zarr_kwargs[ch_name] = new_ch_img
                            input_image_paths.append(new_image_path)
                            new_path_headers.append(ch_name)

                    if verbose:
                        print("Processing {} in {}...".format(file_basename, root))

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
                        applied_crop_size = shape if crop_size is None else crop_size
                        applied_crop_size = tuple(applied_crop_size)

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
                                    break
                            if max_nb_crops_per_image is not None:
                                if slice_idx >= max_nb_crops_per_image:
                                    break

                            # Apply crops:
                            cropped_zarr_kwargs = {zarr_key: data[crop_slc] for zarr_key, data in zarr_kwargs.items()}
                            # Write to zarr file:
                            if save_to_zarr:
                                zarr_utils.append_arrays_to_zarr(out_zarr_path, add_array_dimensions=True, keep_valid_mask=True,
                                                  **cropped_zarr_kwargs)
                            idx_images += 1

                            # Save out image paths:
                            cropped_shape = cropped_zarr_kwargs[main_ch_name].shape
                            new_image_paths.append(
                                input_image_paths + [out_filename, cropped_shape[0], cropped_shape[1]]
                            )

                    else:
                        assert crop_size is None, "When applying crops, image names must be unique"
                        out_name = "{}.png".format(file_basename)
                        if save_to_zarr:
                            zarr_utils.append_arrays_to_zarr(out_zarr_path, add_array_dimensions=True, keep_valid_mask=True,
                                              **zarr_kwargs)
                        idx_images += 1

                        # Save out image paths:
                        cropped_shape = zarr_kwargs[main_ch_name].shape
                        new_image_paths.append(
                            input_image_paths + [out_name, cropped_shape[0], cropped_shape[1]]
                        )
                    new_path_headers += ["Out filename", "shape_x", "shape_y"]
                    path_headers = new_path_headers if path_headers is None else path_headers
                    all_image_paths += new_image_paths

                    # Check if we should stop here:
                    if max_nb_images is not None:
                        if idx_images - starting_index >= max_nb_images:
                            return all_image_paths, idx_images, path_headers

        return all_image_paths, idx_images, path_headers

    all_image_paths, idx_images, path_headers = get_image_paths()
    if verbose:
        print("{} images saved in {}".format(idx_images, out_zarr_path))

    # TODO: add last column with precrop, or offsets

    df = pandas.DataFrame(data=all_image_paths,
                          columns=path_headers)
    if save_to_zarr:
        out_csv_file = out_zarr_path.replace(".zarr", ".csv")

        # if file exists and I am not deleting previous, append:
        if not delete_previous_zarr and os.path.exists(out_csv_file):
            old_df = pandas.read_csv(out_csv_file)
            df = pandas.concat([old_df, df])

        df.to_csv(out_csv_file, index=False)
        return idx_images
    else:
        return df


def convert_multiple_dataset_to_zarr_group(out_zarr_path, channels_list, *multiple_kwargs):
    # Make sure that channel specifications are consistent across all datasets:
    for ch in channels_list:
        for kwargs in multiple_kwargs:
            assert "dataset_name" in kwargs, "Not all dataset names were specified"
            assert ch in kwargs, "Channel specifications for '{}' not found in dataset '{}'".format(
                ch, kwargs["dataset_name"]
            )

    # Process datasets one after the other and add them to the zarr group:
    starting_index = 0
    for i, kwargs  in enumerate(multiple_kwargs):
        kwargs["starting_index"] = starting_index
        kwargs["out_zarr_path"] = out_zarr_path
        if i != 0:
            kwargs["delete_previous_zarr"] = False
        starting_index = convert_images_to_zarr_dataset(**kwargs)
    return starting_index


def apply_preprocessing(preprocessing_function, z_group_path, input_zarr_dataset="BF", out_zarr_dataset=None):
    pass
    #
    # assert os.path.exists(z_group_path), "Zarr file does not exist: {}".format(z_group_path)
    # z_group = zarr.open(z_group_path, mode="w+")
    #
    # # TODO: add option to load all together
    # for i, out_name in enumerate(filenames["Out filename"]):
    #     # Load the dataset:
    #     data = zarr_utils.load_array_from_zarr_group(z_group_path, input_zarr_dataset, z_slice=i)
    #     data = preprocessing_function(data)
    #
    #     # Convert to BGR, where green is ch0 and red is ch1:
    #     img = np.pad(img, pad_width=((0,0), (0,0), (1,1)), mode="constant")
    #
    #     # Get ch1, if needed:
    #     if cellpose_ch1 is not None:
    #         img[...,2] = zarr_utils.load_array_from_zarr_group(z_group_path, cellpose_ch1, apply_valid_mask=True, z_slice=i)
    #
    #     # Write:
    #     cv2.imwrite(os.path.join(out_dir, out_name), img)


def from_zarr_to_cellpose(zarr_group_path, out_dir, cellpose_ch0="BF", cellpose_ch1=None,
                          preprocess_ch0=False, delete_previous=False,
                          enhance_image=False, enhance_factor=3):
    csv_path_path = zarr_group_path.replace(".zarr", ".csv")
    assert os.path.exists(csv_path_path), "No csv file associated to zarr dataset!"

    if delete_previous and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    check_dir_and_create(out_dir)

    filenames = pandas.read_csv(csv_path_path)

    def crop_to_original_size(img, img_specs):
        # Crop to the original size:
        if "shape_x" in img_specs:
            img = img[:img_specs["shape_x"]]
        if "shape_y" in img_specs:
            img = img[:, :img_specs["shape_y"]]
        return img


    for i, out_name in enumerate(filenames["Out filename"]):
        # Get main channel:
        img = zarr_utils.load_array_from_zarr_group(zarr_group_path, cellpose_ch0, z_slice=i)[..., None]
        img = crop_to_original_size(img, filenames.iloc[i])

        if preprocess_ch0:
            img = convert_to_cellpose_style(img, method="subtract")
            # img = preprocess_LIVEcell.preprocess(img)
            # img = img[..., None] if len(img.shape) == 2 else img

        # Convert to BGR, where green is ch0 and red is ch1:
        img = np.pad(img, pad_width=((0,0), (0,0), (1,1)), mode="constant")

        # Get ch1, if needed:
        if cellpose_ch1 is not None:
            ch1 = zarr_utils.load_array_from_zarr_group(zarr_group_path, cellpose_ch1, z_slice=i)
            img[...,2] = crop_to_original_size(ch1, filenames.iloc[i])

        if enhance_image:
            pil_img = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img_out = enhancer.enhance(enhance_factor)
            img = np.array(pil_img_out)



        # Write:
        cv2.imwrite(os.path.join(out_dir, out_name), img)


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
    # process_images_in_path("/scratch/bailoni/datasets/martijn/examplesMacrophages/data",
    #                         "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2", process_channels=True,
    #                        rename_unique=True, max_nb_crops_per_image=1, BF_ch_filter="_c0", DAPI_ch_filter="_c2", delete_out_dir=True)

    # New setup for Veronika images:
    # zarr_out = "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/data_cropped.zarr"
    zarr_out = "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/fixed_data.zarr"
    convert_images_to_zarr_dataset(
        "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/fixed_data",
        zarr_out,
        projectdir_depth=-4,
        # precrop="5400:7600, 5900:9300",
        # max_nb_images=1,
        general_filter="fused_tp_",
        mCherry="_ch_0",
        GFP="_ch_1",
        DAPI="_ch_2",
        # BF1="_ch_3",
        # BF2="_ch_4",
        # BF3="_ch_5",
        verbose=True
    )

    # from_zarr_to_cellpose(zarr_out,
    #                       "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/images",
    #                        cellpose_ch0="BF1",
    #                        cellpose_ch1="DAPI"
    # )

    # from_zarr_to_cellpose(zarr_out,
    #                       "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/images",
    #                       preprocess_ch0=True,
    #                        cellpose_ch0="BF3",
    #                        cellpose_ch1="DAPI"
    # )

    # from_zarr_to_cellpose(zarr_out,
    #                       "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_mCherry/images",
    #                       preprocess_ch0=False,
    #                       cellpose_ch0="mCherry",
    #                       # cellpose_ch1="DAPI"
    #                       )

    # from_zarr_to_cellpose(zarr_out,
    #                       "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/images",
    #                       preprocess_ch0=True,
    #                        cellpose_ch0="BF2",
    #                        cellpose_ch1="DAPI"
    # )


    from_zarr_to_cellpose(zarr_out,
                          "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images_fixed/images",
                           cellpose_ch0="GFP",
                           cellpose_ch1="DAPI"
    )


    print("Done")





