import os.path
import shutil

import zarr
from segmfriends.io.zarr import load_array_from_zarr_group
from segmfriends.io.images import write_segm_to_file
from segmfriends.utils.various import check_dir_and_create
import pandas
import numpy as np


def export_images_from_zarr(main_out_dir,
                            csv_image_data_path,
                            datasets_to_export,
                            filter_main_image_in_csv="_ch_0",
                            delete_previous=False,
                            export_in_external_segm_folder=True,
                            # inner_path,
                            # z_slice=None,
                            # apply_valid_mask=False,
                            # valid_mask_name="valid_mask"
                            ):
    all_image_data = pandas.read_csv(csv_image_data_path)
    assert isinstance(datasets_to_export, (tuple, list))

    if delete_previous and os.path.exists(main_out_dir):
        shutil.rmtree(main_out_dir)

    for dataset in datasets_to_export:
        assert isinstance(dataset, dict)
        for i, image_data in all_image_data.iterrows():
            if "Dataset name" in image_data:
                path_main_img = image_data[2]
                dataset_name = image_data["Dataset name"]
            else:
                path_main_img = image_data[1]
                dataset_name = None
            rel_folder_path, filename = os.path.split(path_main_img)

            filename, extension = os.path.splitext(filename)

            # Get the filter for the main channel:
            if isinstance(filter_main_image_in_csv, dict):
                assert dataset_name is not None
                main_ch_filter = filter_main_image_in_csv[dataset_name]
            else:
                main_ch_filter = filter_main_image_in_csv

            if export_in_external_segm_folder:
                in_dir = os.path.join(main_out_dir, rel_folder_path) if dataset_name is None else os.path.join(main_out_dir, dataset_name, rel_folder_path)
                assert "transformation/cropped_pre_maldi_channels" in in_dir
                out_path = os.path.join(in_dir, "../../cell_segmentation_external/cell_segmentation{}.tif".format(
                    "" if "out_filter" not in dataset else "_" + dataset["out_filter"]))
                out_path = os.path.normpath(out_path)
                check_dir_and_create(os.path.split(out_path)[0])
            else:
                assert main_ch_filter in filename, "Filter '{}' not found in image {}".format(main_ch_filter, path_main_img)
                out_filename = filename.replace(main_ch_filter, dataset["out_filter"])
                out_dir = os.path.join(main_out_dir, rel_folder_path) if dataset_name is None else os.path.join(main_out_dir, dataset_name, rel_folder_path)

                check_dir_and_create(out_dir)
                out_path = os.path.join(out_dir, out_filename + ".tif")
            array = load_array_from_zarr_group(dataset["z_path"],
                                               dataset["inner_path"],
                                               z_slice=i)
            # Apply original crop, if any:
            if "shape_x" in image_data:
                array = array[:image_data["shape_x"]]
            if "shape_y" in image_data:
                array = array[:,:image_data["shape_y"]]
            write_segm_to_file(out_path, array)


if __name__ == "__main__":
    # Test the function:
    export_images_from_zarr("/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/exported_results",
                            "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/data.csv",
                            [{"z_path": "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/predictions/predictions.zarr",
                             "inner_path": "cyto2_diamEst",
                             "out_filter": "_cell_segm"},
                             {
                                 "z_path": "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/predictions/predictions.zarr",
                                 "inner_path": "sem_segmentation",
                                 "out_filter": "_cell_type"},
                             ])
