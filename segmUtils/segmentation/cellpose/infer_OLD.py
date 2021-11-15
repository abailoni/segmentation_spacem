import os
from distutils.dir_util import copy_tree

import shutil

from segmfriends.utils.various import check_dir_and_create

from cellpose import models, io
import cv2
import numpy as np


def multiple_cellpose_inference():
    scratch_dir = "/scratch/bailoni"

    # # Few sample and cropped images:
    # input_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images_small/cellpose")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1_small_images")

    # # Images from Alyona:
    # input_dir = os.path.join(scratch_dir, "projects/train_cellpose/data/test")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1")

    models_to_test = {
        # "finetuned_LIVECell_lr_02": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_17_10_26_25.124850",
        "cyto2": "cyto2",
        # "trained_on_LIVECell": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_11_23_44_58.553246",
        # "cyto": "cyto",
        # "trained_on_cellpose": "/scratch/bailoni/datasets/cellpose/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_14_21_14_37.300226",
        # "finetuned_LIVECell_lr_00002": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_17_10_32_40.166405",
        # "cleaned_LIVECell": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_17_28_59.152864",
        # "cleaned_LIVECell_pip": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_22_50_40.020632",
        # "cleaned_finetuned_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_08_49.114912",
        # "cleaned_from_scratch_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_02_49.688436",
        # "scratch": None
        # "full_LIVECell_lr_002_SGD_cyto2": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_2021_10_25_15_19_08.80932",

    }

    overwrite_estimate_diameter = {
        "cyto2": True
    }

    dirs_to_process = [
        # # # LIVECell, CellPose full, Alex:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/alex_labeled/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alex")
        # ],
        # [
        #     os.path.join(scratch_dir, "datasets/LIVECell/panoptic/livecell_coco_test_cleaned"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/LIVECell_test_cleaned")
        # ],
        # [
        #     os.path.join(scratch_dir, "datasets/cellpose/test"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/cellpose_test")
        # ],
        # ------------------------------------------------
        # Few cropped images:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/input_images_small/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/cropped_images")
        # ],
        # ------------------------------------------------
        # Alyona images:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/input_images/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alyona")
        # ],
        # ------------------------------------------------
        # New images from Alex:
        # [
        #     "/scratch/bailoni/datasets/alex/glioblastoma/preprocessed/cellpose",
        #     "/scratch/bailoni/datasets/alex/glioblastoma/segmentations/$MODEL_NAME"
        # ]
        # ------------------------------------------------
        # New datasets from Alex and Martjin
        # [
        #     "/scratch/bailoni/datasets/alex/glioblastoma-v2/preprocessed/cellpose",
        #     "/scratch/bailoni/datasets/alex/glioblastoma-v2/segmentations/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2/cellpose",
        #     "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2/$MODEL_NAME"
        # ],
        ######### Veronika:
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/predictions/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF2_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF2_DAPI/predictions/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/predictions/$MODEL_NAME"
        # ],
        [
            "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/images",
            "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/predictions/$MODEL_NAME"
        ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/predictions/$MODEL_NAME"
        # ],
    ]

    # diameter = 30

    save_npy = False
    estimate_diameter = [False]

    first_ch = 2
    second_ch = 1
    mask_filter = "_masks"

    for default_est_diam in estimate_diameter:
        for model_name, model_path in models_to_test.items():
            est_diam = overwrite_estimate_diameter[
                model_name] if model_name in overwrite_estimate_diameter else default_est_diam
            diameter = 0 if est_diam else 30
            # Update model names accordingly:
            model_name = model_name + "_diamEst" if est_diam else model_name + "_noDiamEst"

            for input_dir, out_dir in dirs_to_process:
                out_dir = out_dir.replace("$MODEL_NAME", model_name)
                # Create model directory:
                check_dir_and_create(os.path.dirname(os.path.normpath(out_dir)))
                # Create out dir:
                check_dir_and_create(out_dir)

                # TODO: create function
                # TODO: need to specify ipython path for debugging with PyCharm
                command = "/scratch/bailoni/miniconda3/envs/pyT17/bin/ipython -m cellpose " \
                          "-- --use_gpu --dir {} --savedir {} --pretrained_model {} " \
                          "--chan {} --chan2 {} " \
                          "--mask_filter {} --save_png --diameter {} " \
                          "--use_size_model {}".format(
                    input_dir,
                    out_dir,
                    model_path,
                    first_ch,
                    second_ch,
                    mask_filter,
                    diameter,
                    "" if save_npy else "--no_npy"
                )
                os.system(command)




if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"

    # # Few sample and cropped images:
    # input_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images_small/cellpose")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1_small_images")

    # # Images from Alyona:
    # input_dir = os.path.join(scratch_dir, "projects/train_cellpose/data/test")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1")

    models_to_test = {
        # "finetuned_LIVECell_lr_02": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_17_10_26_25.124850",
        "cyto2": "cyto2",
        # "trained_on_LIVECell": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_11_23_44_58.553246",
        # "cyto": "cyto",
        # "trained_on_cellpose": "/scratch/bailoni/datasets/cellpose/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_14_21_14_37.300226",
        # "finetuned_LIVECell_lr_00002": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_17_10_32_40.166405",
        # "cleaned_LIVECell": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_17_28_59.152864",
        # "cleaned_LIVECell_pip": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_22_50_40.020632",
        # "cleaned_finetuned_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_08_49.114912",
        # "cleaned_from_scratch_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_02_49.688436",
        # "scratch": None
        # "full_LIVECell_lr_002_SGD_cyto2": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_2021_10_25_15_19_08.80932",

    }

    overwrite_estimate_diameter = {
        "cyto2": True
    }


    dirs_to_process = [
        # # # LIVECell, CellPose full, Alex:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/alex_labeled/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alex")
        # ],
        # [
        #     os.path.join(scratch_dir, "datasets/LIVECell/panoptic/livecell_coco_test_cleaned"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/LIVECell_test_cleaned")
        # ],
        # [
        #     os.path.join(scratch_dir, "datasets/cellpose/test"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/cellpose_test")
        # ],
        # ------------------------------------------------
        # Few cropped images:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/input_images_small/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/cropped_images")
        # ],
        # ------------------------------------------------
        # Alyona images:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/input_images/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alyona")
        # ],
        # ------------------------------------------------
        # New images from Alex:
        # [
        #     "/scratch/bailoni/datasets/alex/glioblastoma/preprocessed/cellpose",
        #     "/scratch/bailoni/datasets/alex/glioblastoma/segmentations/$MODEL_NAME"
        # ]
        # ------------------------------------------------
        # New datasets from Alex and Martjin
        # [
        #     "/scratch/bailoni/datasets/alex/glioblastoma-v2/preprocessed/cellpose",
        #     "/scratch/bailoni/datasets/alex/glioblastoma-v2/segmentations/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2/cellpose",
        #     "/scratch/bailoni/datasets/martijn/examplesMacrophages/preprocessed_BR_ch2/$MODEL_NAME"
        # ],
        ######### Veronika:
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF1_DAPI/predictions/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF2_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_BF2_DAPI/predictions/$MODEL_NAME"
        # ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_procBF3_DAPI/predictions/$MODEL_NAME"
        # ],
        [
            "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/images",
            "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_GFP_DAPI_full_images/predictions/$MODEL_NAME"
        ],
        # [
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/images",
        #     "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6/cellpose_LCprocBF2_DAPI/predictions/$MODEL_NAME"
        # ],
    ]


    # diameter = 30

    save_npy = False
    estimate_diameter = [False]



    first_ch = 2
    second_ch = 1
    mask_filter = "_masks"

    for default_est_diam in estimate_diameter:
        for model_name, model_path in models_to_test.items():
            est_diam = overwrite_estimate_diameter[model_name] if model_name in overwrite_estimate_diameter else default_est_diam
            diameter = 0 if est_diam else 30
            # Update model names accordingly:
            model_name = model_name + "_diamEst" if est_diam else model_name + "_noDiamEst"

            for input_dir, out_dir in dirs_to_process:
                out_dir = out_dir.replace("$MODEL_NAME", model_name)
                # Create model directory:
                check_dir_and_create(os.path.dirname(os.path.normpath(out_dir)))
                # Create out dir:
                check_dir_and_create(out_dir)

                # TODO: create function
                # TODO: need to specify ipython path for debugging with PyCharm
                command = "/scratch/bailoni/miniconda3/envs/pyT17/bin/ipython -m cellpose " \
                          "-- --use_gpu --dir {} --savedir {} --pretrained_model {} " \
                          "--chan {} --chan2 {} " \
                          "--mask_filter {} --save_png --diameter {} " \
                          "--use_size_model {}".format(
                    input_dir,
                    out_dir,
                    model_path,
                    first_ch,
                    second_ch,
                    mask_filter,
                    diameter,
                    "" if save_npy else "--no_npy"
                )
                os.system(command)

