import os
from distutils.dir_util import copy_tree

import shutil

from segmfriends.utils.various import check_dir_and_create

from cellpose import models, io
import cv2
import numpy as np


def infer_cellpose_directory(in_dir, out_dir, keep_input_images_in_out_dir=False):
    # By default, CellPose outputs stuff in the same folder.
    # TODO: can be easily changed with the --savedir option...
    # To avoid that, we copy images to the output folder and then delete them
    # FIXME: if the output is not empty, this makes a mess (read them as inputs, and delete them afterwards)
    assert not keep_input_images_in_out_dir, "Not implemented"
    check_dir_and_create(out_dir)
    shutil.rmtree(out_dir)
    check_dir_and_create(out_dir)
    copy_tree(in_dir, out_dir)

    input_files = []
    for root, dirs, files in os.walk(out_dir):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
                if not filename.startswith("."):
                    input_files.append([filename, root])



    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=True, model_type='cyto2')

    for filename, root in input_files:
        image = cv2.imread(os.path.join(root, filename))
        masks, flows, styles, diams = model.eval(image, diameter=None, channels=[2,0])

        _, file_extension = os.path.splitext(filename)

        # Plot result:
        import matplotlib.pyplot as plt
        from segmfriends.vis import plot_segm, get_figure, save_plot
        fig, ax = get_figure(1,1, figsize=(15,15))
        gray_img = image[...,1][None]
        plot_segm(ax, masks[None], background=gray_img, mask_value=0)
        save_plot(fig, root, filename.replace(file_extension, "_out_plot{}".format(file_extension)))
        plt.close(fig)

        cv2.imwrite(os.path.join(root, filename.replace(file_extension, "_segm{}".format(file_extension))),
                    masks.astype(np.uint16))


    # command = "python -m cellpose --dir {} --pretrained_model cyto2 --chan 2 --chan2 1 --use_gpu".format( # --no_npy --save_png
    #     out_dir
    # )
    # os.system(command)
    # stream = os.popen(command)
    # print(stream.read())



    # If needed, remove original image files:
    if not keep_input_images_in_out_dir:
        for filename, root in input_files:
            os.remove(os.path.join(root, filename))

    #




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
        # "cyto2": "cyto2",
        # "trained_on_LIVECell": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_11_23_44_58.553246",
        # "cyto": "cyto",
        # "trained_on_cellpose": "/scratch/bailoni/datasets/cellpose/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_14_21_14_37.300226",
        # "finetuned_LIVECell_lr_00002": "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_17_10_32_40.166405",
        # "cleaned_LIVECell": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_17_28_59.152864",
        # "cleaned_LIVECell_pip": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_21_22_50_40.020632",
        "cleaned_finetuned_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_08_49.114912",
        "cleaned_from_scratch_LIVECell_v1": "/scratch/bailoni/datasets/LIVECell/panoptic/livecell_coco_train_cleaned/models/cellpose_residual_on_style_on_concatenation_off_livecell_coco_train_cleaned_2021_10_22_17_02_49.688436",

        # "scratch": None
    }


    dirs_to_process = [
        # # # LIVECell, CellPose full, Alex:
        # [
        #     os.path.join(scratch_dir, "projects/spacem_segm/alex_labeled/cellpose"),
        #     os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alex")
        # ],
        [
            os.path.join(scratch_dir, "datasets/LIVECell/panoptic/livecell_coco_test_cleaned"),
            os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/LIVECell_test_cleaned")
        ],
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
    ]


    # diameter = 30





    first_ch = 2
    second_ch = 1
    mask_filter = "_masks"

    for diameter in [0]:
        for model_name, model_path in models_to_test.items():
            # Update model names accordingly:
            model_name = model_name + "_diamEst" if diameter == 0 else model_name + "_noDiamEst"

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
                          "--no_npy --mask_filter {} --save_png --diameter {} " \
                          "--use_size_model".format(
                    input_dir,
                    out_dir,
                    model_path,
                    first_ch,
                    second_ch,
                    mask_filter,
                    diameter
                )
                os.system(command)

