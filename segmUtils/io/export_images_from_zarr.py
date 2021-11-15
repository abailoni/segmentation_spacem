import os.path

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
                            # inner_path,
                            # z_slice=None,
                            # apply_valid_mask=False,
                            # valid_mask_name="valid_mask"
                            ):
    all_image_data = pandas.read_csv(csv_image_data_path)
    assert isinstance(datasets_to_export, (tuple, list))

    for dataset in datasets_to_export:
        assert isinstance(dataset, dict)
        for i, image_data in all_image_data.iterrows():
            path_main_img = image_data[1]
            rel_folder_path, filename = os.path.split(path_main_img)

            filename, extension = os.path.splitext(filename)
            assert filter_main_image_in_csv in filename, "Filter '{}' not found in image {}".format(filter_main_image_in_csv, path_main_img)
            out_filename = filename.replace(filter_main_image_in_csv, dataset["out_filter"])
            out_dir = os.path.join(main_out_dir, rel_folder_path)
            check_dir_and_create(out_dir)
            out_path = os.path.join(out_dir, out_filename + ".tif")
            array = load_array_from_zarr_group(dataset["z_path"],
                                               dataset["inner_path"],
                                               z_slice=i)
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
