import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import imageio
import shutil
from copy import deepcopy
import pandas as pd
import vigra

from segmfriends.utils.various import check_dir_and_create
import segmfriends.utils.various as segm_utils
import segmfriends.vis as segm_vis

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rc('font',family='sans-serif', size=8)
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'


def auto_pad_arrays_to_same_shape(arrays, pad_value=0, allow_rotations=False):
    max_shape = None
    assert len(arrays) > 0

    # Find max shape and eventually rotate arrays
    for indx, arr in enumerate(arrays):
        arr_shape = arr.shape
        assert len(arr_shape) == 2, "Only 2D images supported atm"
        if max_shape is None:
            max_shape = arr_shape
        else:
            max_shape_diff = [0 if max_shp >= arr_shape[i] else (arr_shape[i] - max_shp) for i, max_shp in
                              enumerate(max_shape)]
            # Now try by rotating image:
            max_shape_diff_rot = [0 if max_shp >= arr_shape[1-i] else (arr_shape[1-i] - max_shp) for i, max_shp in
                             enumerate(max_shape)]
            # Check which one requires less padding:
            diff, diff_rot = np.array(max_shape_diff).sum(), np.array(max_shape_diff_rot).sum()

            rotate_image = False
            if allow_rotations:
                rotate_image = diff_rot < diff
                if rotate_image:
                    arrays[indx] = np.transpose(arr)

            selected_diff = max_shape_diff_rot if rotate_image else max_shape_diff
            # Now update the maximum shape:
            max_shape = [dif + max_shp  for max_shp, dif in zip(max_shape, selected_diff)]

    # Now pad the arrays to the maximum shape:
    for indx, arr in enumerate(arrays):
        if tuple(max_shape) != arr.shape:
            shape_diff = [max_shp - img_shp for max_shp, img_shp in
                          zip(max_shape, arr.shape)]
            assert all([shp >= 0 for shp in shape_diff]), "Something went wrong with image zero padding"
            arrays[indx] = np.pad(arr, pad_width=((0, shape_diff[0]), (0, shape_diff[1])), constant_values=pad_value)

    return arrays

def convert_dataset_to_hdf5_volume(raw_dir, GT_dir, out_h5_path,
                           raw_filter="", GT_filter="_masks",
                                   raw_extension=".png",
                                   GT_extension=".png", raw_channel=1):
    assert os.path.exists(raw_dir)
    assert os.path.exists(GT_dir)
    assert isinstance(raw_channel, int)

    raw_collected = []
    GT_collected = []
    max_label = 0
    for root, dirs, files in os.walk(raw_dir):
        for filename in sorted(files):
            file_basename, file_extension = os.path.splitext(filename)
            if file_basename.endswith(raw_filter) and file_extension == raw_extension and not file_basename.endswith(GT_filter):
                # Look for corresponding GT masks:
                if raw_filter != "":
                    GT_basename = file_basename.replace(raw_filter, GT_filter)
                else:
                    GT_basename = file_basename + GT_filter
                GT_masks_path = os.path.join(GT_dir, GT_basename + GT_extension)
                assert os.path.exists(GT_masks_path), "GT file not found!"

                # Load images:
                raw = cv2.imread(os.path.join(root, filename))
                raw_collected.append(raw[:, :, raw_channel])  # Only keep green channel

                GT_segm = imageio.imread(GT_masks_path).astype('uint32')
                GT_segm[GT_segm != 0] += max_label
                max_label = GT_segm.max() + 1
                GT_collected.append(GT_segm)

        # Only explore first directory:
        break

    assert len(raw_collected) > 0, "No images found in given folder"
    # Concatenate and save:
    raw_collected = np.stack(auto_pad_arrays_to_same_shape(raw_collected, allow_rotations=True), axis=0)
    GT_collected = np.stack(auto_pad_arrays_to_same_shape(GT_collected, allow_rotations=True), axis=0)
    segm_utils.writeHDF5(raw_collected, out_h5_path, "raw")
    segm_utils.writeHDF5(GT_collected, out_h5_path, "gt")


def convert_segmentations_to_hdf5_volume(pred_dir, out_h5_path,
                                         inner_h5_path="data",
                                   pred_filter="_masks",
                                   pred_extension=".png"):
    assert os.path.exists(pred_dir)
    pred_collected = []
    max_label = 0
    for root, dirs, files in os.walk(pred_dir):
        for filename in sorted(files):
            file_basename, file_extension = os.path.splitext(filename)
            if file_basename.endswith(pred_filter) and file_extension == pred_extension:
                # Load segmentation:
                pred_segm = imageio.imread(os.path.join(root, filename)).astype('uint32')
                pred_segm[pred_segm != 0] += max_label
                max_label = pred_segm.max() + 1
                pred_collected.append(pred_segm)



        # Only explore first directory:
        break

    assert len(pred_collected) > 0, "No segmentations found in given folder"
    # Concatenate and save:
    pred_collected = np.stack(auto_pad_arrays_to_same_shape(pred_collected, allow_rotations=True), axis=0)
    segm_utils.writeHDF5(pred_collected, out_h5_path, inner_h5_path)



if __name__ == "__main__":
    # -----------------------
    # Parameters to choose:
    # -----------------------
    datasets_to_convert = [
        # "alex",
        # "LIVECell_train",
        "LIVECell_test",
    ]

    model_predictions_to_convert = {
        # "cyto2_diamEst": "CellPose cyto2 model",
                      # "cyto_diamEst": "CellPose cyto",
                      # "trained_on_LIVECell_noDiamEst": "Trained on LIVECell+CellPose from scratch",
                      # "trained_on_LIVECell_diamEst": "Trained on LIVECell+CellPose data (est diam)",
                      # "trained_on_cellpose_noDiamEst": "Trained on CellPose data",
                      # "trained_on_cellpose_diamEst": "Trained on CellPose data (est diam)",
                      # "finetuned_LIVECell_lr_02_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_diamEst": "finetuned_LIVECell_lr_00002_diamEst",
                      # "finetuned_LIVECell_lr_02_diamEst": "finetuned_LIVECell_lr_02_diamEst",
                      }

    OUT_DIR = "/scratch/bailoni/projects/train_cellpose/hdf5_data"
    # -----------------------

    RAW_IMAGE_DIR = {
        "LIVECell_train": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train",
        "LIVECell_test": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned",
        "cellpose_test": "/scratch/bailoni/datasets/cellpose/test",
        "alex": "/scratch/bailoni/projects/spacem_segm/alex_labeled/cellpose"
    }
    GT_DIR = {
        "LIVECell_train": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train",
        "LIVECell_test": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned",
        "cellpose_test": "/scratch/bailoni/datasets/cellpose/test",
        "alex": "/scratch/bailoni/datasets/alex/labels"
    }

    PREDICTION_DIR = "/scratch/bailoni/projects/train_cellpose/predictions"

    check_dir_and_create(OUT_DIR)


    for dataset_name in datasets_to_convert:
            segm_utils.check_dir_and_create(os.path.join(OUT_DIR, dataset_name))
            convert_dataset_to_hdf5_volume(
                raw_dir=RAW_IMAGE_DIR[dataset_name],
                GT_dir=GT_DIR[dataset_name],
                out_h5_path=os.path.join(OUT_DIR, dataset_name, "raw_and_gt.h5")
            )

            for model_idx, model_name in enumerate(model_predictions_to_convert):
                convert_segmentations_to_hdf5_volume(
                    pred_dir=os.path.join(PREDICTION_DIR, model_name, dataset_name),
                    out_h5_path=os.path.join(OUT_DIR, dataset_name, model_name + ".h5")
                )



