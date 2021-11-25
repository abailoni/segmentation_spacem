import shutil
from copy import deepcopy
import os
import sys

import json
import numpy as np
import pandas

from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmUtils.postprocessing.compute_scores import compute_scores
from segmfriends.speedrun_exps.utils import process_speedrun_sys_argv
from segmfriends.utils import check_dir_and_create

from speedrun import BaseExperiment
from segmfriends.utils.paths import get_vars_from_argv_and_pop

from .base_experiment import CellposeBaseExperiment, CoreSpaceMExperiment
from .train import start_cellpose_training

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, convert_multiple_cellpose_output_to_zarr


def collect_images_from_multiple_datasets(train_dir, datasets_dirs,
                                          delete_previous=False,
                                          # cellpose_raw_filter="",
                                          # cellpose_mask_filter="_masks"
                                          ):
    # TODO: if something was already there (npy needs to be recomputed every time otherwise...)
    if os.path.exists(train_dir) and delete_previous:
        shutil.rmtree((train_dir))
    check_dir_and_create(train_dir)

    for data_dir in datasets_dirs:
        rsync_command = "rsync -a {} {}".format(os.path.join(data_dir, "*"),
                                      train_dir)
        out = os.system(rsync_command)
        assert out == 0, "Something went wrong while copying the data"

        # raw_config = dataset_config["raw"]
        # GT_config = dataset_config["GT"]
        # for root, dirs, files in os.walk(raw_config["dir"]):
        #     for raw_filename in files:
        #         file_basename, raw_extension = os.path.splitext(raw_filename)
        #         # Check extension and filters:
        #         if raw_extension != raw_config.get("extension"):
        #             continue
        #         if not file_basename.endswith(raw_config.get("filter")):
        #             continue
        #
        #         # Get GT path:
        #         if raw_config.get("filter") == "":
        #             out_raw_filename = file_basename + cellpose_raw_filter + file
        #             GT_filename = file_basename + GT_config["filter"] + GT_config["extension"]
        #         else:
        #             GT_filename = file_basename.replace(raw_config["filter"], GT_config["filter"]) + GT_config["extension"]
        #
        #         # Copy to train directory, if not already there
        #         GT_path = os.path
        #     # Explore only top directory:
        #     break

    # TODO: check that no previous leftovers images are left in the train_dir (if I do not delete everything at the
    #  beginning) Problem: if I update images, .npy files won.t be recomputed...?

class TrainingCellposeExperiment(CellposeBaseExperiment):
    def train_cellpose(self):
        # Replace possible $EXP_PATH placeholders:
        self.replace_experiment_placeholder_in_config("train_cellpose")

        # # Get dataset configs:
        # list_used_datasets = self.get("train_cellpose/datasets", ensure_exists=True)
        # global_config = self.get("train_cellpose/global_config", {})
        #
        # all_configs = []
        # for dataset in list_used_datasets:
        #     config = deepcopy(global_config)
        #     config.update(self.get("train_cellpose/{}".format(dataset), {}))
        #     all_configs.append(config)

        # Copy all the training images from different datasets in one single folder:
        train_dir = self.get("train_cellpose/main_train_dir")
        input_train_dirs = self.get("train_cellpose/input_train_dirs")
        if input_train_dirs is not None:
            collect_images_from_multiple_datasets(train_dir, input_train_dirs)
        val_dir = self.get("train_cellpose/main_val_dir")
        input_val_dirs = self.get("train_cellpose/input_val_dirs")
        if input_val_dirs is not None:
            collect_images_from_multiple_datasets(train_dir, input_val_dirs)

        # Start cellpose training:
        cellpose_training_kwargs = self.get("train_cellpose/cellpose_training_kwargs", {})
        cellpose_training_args = self.get("train_cellpose/cellpose_training_args", [])

        # To avoid confusion, delete any model that was trained previously:
        model_directory = os.path.join(self.experiment_directory,
                     "Weights")
        shutil.rmtree(model_directory)
        check_dir_and_create(model_directory)
        cellpose_training_kwargs["save_path"] = model_directory
        start_cellpose_training(train_dir, val_dir,
                                *cellpose_training_args,
                                **cellpose_training_kwargs)

