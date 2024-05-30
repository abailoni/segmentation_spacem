import numpy as np

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

import json

from shutil import copy




def copy_clean_images(in_raw_dir, out_raw_dir, in_GT_dir, out_GT_dir, clean_image_mask,
                           raw_filter="", GT_filter="_masks",
                                   raw_extension=".png",
                                   GT_extension=".png"):
    assert os.path.exists(in_raw_dir)
    assert os.path.exists(in_GT_dir)

    check_dir_and_create(out_raw_dir)
    shutil.rmtree(out_raw_dir)
    check_dir_and_create(out_raw_dir)

    check_dir_and_create(out_GT_dir)
    shutil.rmtree(out_GT_dir)
    check_dir_and_create(out_GT_dir)

    indx = 0
    for root, dirs, files in os.walk(in_raw_dir):
        for filename in sorted(files):
            file_basename, file_extension = os.path.splitext(filename)
            if file_basename.endswith(raw_filter) and file_extension == raw_extension and not file_basename.endswith(GT_filter):
                # Look for corresponding GT masks:
                if raw_filter != "":
                    GT_basename = file_basename.replace(raw_filter, GT_filter)
                else:
                    GT_basename = file_basename + GT_filter
                GT_segm_path = os.path.join(in_GT_dir, GT_basename + GT_extension)
                assert os.path.exists(GT_segm_path), "GT file not found!"

                # Copy clean images:
                if clean_image_mask[indx]:
                    raw_path = os.path.join(root, filename)
                    copy(raw_path, out_raw_dir)
                    copy(GT_segm_path, out_GT_dir)

                indx += 1

        # Only explore first directory:
        break


def get_z_indices_from_json_dict(annotations, max_index, keep_selected=True, coordinate_indx=0):
    selected_coords = []
    for ann in annotations:
        if int(ann["point"][coordinate_indx]) <= max_index:
            selected_coords.append(int(ann["point"][coordinate_indx]))
    selected_coords = np.array(selected_coords)

    # if max_index is None:
    #     assert keep_selected, "Max index needed for inverting the selection"
    #     max_index = selected_coords.max()
    selected = np.zeros((max_index+1,), dtype="bool")
    selected[selected_coords] = True

    if not keep_selected:
        selected = np.logical_not(selected)

    return selected


if __name__ == "__main__":
    # -----------------------
    # Parameters to choose:
    # -----------------------
    datasets_to_convert = [
        # "alex",
        "LIVECell_test"
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

    JSON_files = {
        "LIVECell_train": "/scratch/bailoni/pyCh_repos/segmentation_spacem/configs/LIVECell_clean_annotations_train.json",
        "LIVECell_test": "/scratch/bailoni/pyCh_repos/segmentation_spacem/configs/LIVECell_clean_annotations_test.json",
    }

    RAW_IMAGE_DIR = {
        "LIVECell_train": ["/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train",
                           "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned"],
        "LIVECell_test": ["/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned_temp",
                           "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned"],
        # "LIVECell_test": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test",
        # "cellpose_test": "/scratch/bailoni/datasets/cellpose/test",
        # "alex": "/scratch/bailoni/projects/spacem_segm/alex_labeled/cellpose"
    }
    GT_DIR = {
        "LIVECell_test": ["/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned_temp",
                          "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test_cleaned"],
        # "LIVECell_test": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_test",
        # "cellpose_test": "/scratch/bailoni/datasets/cellpose/test",
        # "alex": "/scratch/bailoni/datasets/alex/labels"
    }

    PREDICTION_DIR = "/scratch/bailoni/projects/train_cellpose/predictions"

    check_dir_and_create(OUT_DIR)


    for dataset_name in datasets_to_convert:
        with open(JSON_files[dataset_name]) as f:
            anno = json.load(f)

        annotations = anno["layers"][0]["annotations"]

        if dataset_name == "LIVECell_train":
            part1 = get_z_indices_from_json_dict(annotations, 1499, keep_selected=False)
            part2 = get_z_indices_from_json_dict(annotations, 3188, keep_selected=True)

            images_to_keep = part2
            images_to_keep[:1500] = part1
        elif dataset_name == "LIVECell_test":
            images_to_keep = get_z_indices_from_json_dict(annotations, 1359, keep_selected=True)
        else:
            raise NotImplementedError

        copy_clean_images(
            RAW_IMAGE_DIR[dataset_name][0],
            RAW_IMAGE_DIR[dataset_name][1],
            GT_DIR[dataset_name][0],
            GT_DIR[dataset_name][1],
            images_to_keep
        )
