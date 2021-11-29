import os.path
import sys

import numpy as np
import vigra
import segmfriends.io.zarr as zarr_utils
import nifty.graph.rag
import nifty.tools as ntools

from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmfriends.speedrun_exps.utils import process_speedrun_sys_argv

from speedrun import BaseExperiment
from segmfriends.utils.paths import get_vars_from_argv_and_pop

from segmUtils.segmentation.cellpose.train_experiment import TrainingCellposeExperiment

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, convert_multiple_cellpose_output_to_zarr


if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path, default_config_rel_path="./configs/train",
         default_exp_path="/scratch/bailoni/projects/cellpose_projects/finetuning/train")

    cls = TrainingCellposeExperiment
    cls().run()
