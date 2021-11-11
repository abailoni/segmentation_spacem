import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import imageio
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

import shutil
from copy import deepcopy
import pandas as pd
import vigra

from segmfriends.utils.various import check_dir_and_create
import segmfriends.utils.various as segm_utils
import segmfriends.vis as segm_vis

from segmfriends.io.zarr import append_arrays_to_zarr

def convert_segmentations_to_zarr(pred_dir, out_zarr_path,
                                         model_name,
                                         pred_filter="_masks",
                                         pred_extension=".png"):
    # FIXME: make sure that order is the same (at the moment it works with sorting)
    assert os.path.exists(pred_dir)
    saved_images = 0
    max_label = 0
    for root, dirs, files in os.walk(pred_dir):
        for filename in sorted(files):
            file_basename, file_extension = os.path.splitext(filename)
            if file_basename.endswith(pred_filter) and file_extension == pred_extension:
                # Load segmentation:
                pred_segm = imageio.imread(os.path.join(root, filename)).astype('uint32')
                pred_segm[pred_segm != 0] += max_label
                max_label = pred_segm.max() + 1
                append_arrays_to_zarr(out_zarr_path, add_array_dimensions=True,
                                      **{model_name: pred_segm}
                                      )
                saved_images += 1
        # Only explore first directory:
        break

    assert saved_images, "No segmentations found in given folder"



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



