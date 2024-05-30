import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

import os
import json, cv2, random
import imageio
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

import shutil
from copy import deepcopy
import pandas as pd

from segmfriends.utils.various import check_dir_and_create
import segmfriends.utils.various as segm_utils
import segmfriends.vis as segm_vis

from segmfriends.io.zarr import append_arrays_to_zarr
import pandas

def convert_segmentations_to_zarr(pred_dir, out_zarr_path,
                                         model_name,
                                        csv_file_path,
                                         pred_filter="_masks",
                                         pred_extension=".png"):
    assert os.path.exists(pred_dir)
    saved_images = 0
    max_label = 0

    # for root, dirs, files in os.walk(pred_dir):
    #     for filename in sorted(files):
    filenames = pandas.read_csv(csv_file_path)
    for i, filename in enumerate(filenames["Out filename"]):
        file_basename, file_extension = os.path.splitext(filename)
        # FIXME: generalize and use pred_filter instead
        pred_name = file_basename + "_cp_masks.png"
        print(pred_name)
        pred_path = os.path.join(pred_dir, pred_name)
        assert os.path.exists(pred_path), pred_path
        pred_segm = imageio.imread(pred_path).astype('uint32')
        pred_segm[pred_segm != 0] += max_label
        max_label = pred_segm.max() + 1
        append_arrays_to_zarr(out_zarr_path, add_array_dimensions=True,
                              **{model_name: pred_segm}
                              )
        saved_images += 1

    assert saved_images, "No segmentations found in given folder"

def convert_multiple_cellpose_output_to_zarr(main_pred_directory,
                                             csv_file,
                                             delete_previous=True):
    zarr_filename = "predictions_collected.zarr"
    out_zarr_group_path = os.path.join(main_pred_directory, zarr_filename)

    if delete_previous and os.path.exists(out_zarr_group_path):
        shutil.rmtree(out_zarr_group_path)

    collected_model_names = []
    for root, dirs, files in os.walk(main_pred_directory):
        for model_name in dirs:
            if model_name != zarr_filename:
                convert_segmentations_to_zarr(
                    pred_dir=os.path.join(main_pred_directory, model_name),
                    out_zarr_path=out_zarr_group_path,
                    model_name=model_name,
                    csv_file_path=csv_file
                )
            collected_model_names.append(model_name)
        # Only check the directory top-level:
        break
    return out_zarr_group_path, collected_model_names


if __name__ == "__main__":
    # -----------------------
    # Parameters to choose:
    # -----------------------
    # datasets_to_convert = [
    #     # "alex",
    #     # "LIVECell_train",
    #     "LIVECell_test",
    # ]

    model_predictions_to_convert = [
        "cyto2_diamEst",
        # "cleaned_finetuned_LIVECell_v1_noDiamEst"
                      # "cyto_diamEst": "CellPose cyto",
                      # "trained_on_LIVECell_noDiamEst": "Trained on LIVECell+CellPose from scratch",
                      # "trained_on_LIVECell_diamEst": "Trained on LIVECell+CellPose data (est diam)",
                      # "trained_on_cellpose_noDiamEst": "Trained on CellPose data",
                      # "trained_on_cellpose_diamEst": "Trained on CellPose data (est diam)",
                      # "finetuned_LIVECell_lr_02_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_diamEst": "finetuned_LIVECell_lr_00002_diamEst",
                      # "finetuned_LIVECell_lr_02_diamEst": "finetuned_LIVECell_lr_02_diamEst",
                      ]

    # OUT_DIR = "/scratch/bailoni/projects/train_cellpose/hdf5_data"
    # -----------------------

    main_pred_dir = "/scratch/bailoni/projects/train_cellpose/predictions"
    PREDICTION_DIRS = [
        # "/scratch/bailoni/datasets/martijn/examplesMacrophages/predictions_BF",
        # "/scratch/bailoni/datasets/martijn/examplesMacrophages/predictions_ch2",
        # "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2",
        # "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/predictions/",
        # "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF2_DAPI/predictions/",
        # "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF1_DAPI/predictions/",
        # "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/predictions/",
        # "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/predictions/",
        "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/predictions/",
    ]


    for pred_dir in PREDICTION_DIRS:
        for model_name in model_predictions_to_convert:
            model_pred_dir = os.path.join(pred_dir, model_name)
            assert os.path.exists(model_pred_dir)
            convert_segmentations_to_zarr(
                pred_dir=model_pred_dir,
                out_zarr_path=os.path.join(pred_dir, "predictions.zarr"),
                model_name=model_name
            )



