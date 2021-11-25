import os
from distutils.dir_util import copy_tree

import shutil

from segmfriends.utils.various import check_dir_and_create

from cellpose import models, io
import cv2
import numpy as np



def multiple_cellpose_inference(tested_models,
                                overwrite_estimate_diameter,
                                dirs_to_process,
                                estimate_diameter=False,
                                cellpose_infer_args=( ),
                                **cellpose_infer_kwargs
                                ):
    estimate_diameter = estimate_diameter if isinstance(estimate_diameter, (tuple, list)) else [estimate_diameter]

    for default_est_diam in estimate_diameter:
        for model_name, model_path in tested_models.items():
            est_diam = overwrite_estimate_diameter[
                model_name] if model_name in overwrite_estimate_diameter else default_est_diam
            diameter = 0 if est_diam else 30
            # Update model names accordingly:
            model_name = model_name + "_diamEst" if est_diam else model_name + "_noDiamEst"

            for input_dir, out_dir in dirs_to_process:
                out_dir = out_dir.replace("$MODEL_NAME", model_name)
                # Delete previous predictions:
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                # Create model directory:
                check_dir_and_create(os.path.dirname(os.path.normpath(out_dir)))
                # Create out dir:
                check_dir_and_create(out_dir)

                # TODO: find a way to specify ipython path for debugging with PyCharm
                command = "/scratch/bailoni/miniconda3/envs/pyT17/bin/ipython -m cellpose " \
                          "-- --use_gpu --dir {} --savedir {} --pretrained_model {} " \
                          "--save_png --diameter {} " \
                          "--use_size_model ".format(
                    input_dir,
                    out_dir,
                    model_path,
                    diameter
                )

                # Add the args:
                for arg in cellpose_infer_args:
                    assert isinstance(arg, str), "Arguments should be strings"
                    command += "--{} ".format(arg)

                # Add the kwargs:
                for kwarg in cellpose_infer_kwargs:
                    command += "--{} {} ".format(kwarg, cellpose_infer_kwargs[kwarg])
                print(command)
                os.system(command)
