import os.path
import sys

from speedrun import process_speedrun_sys_argv

from segmUtils.segmentation.cellpose.base_experiment import CellposeBaseExperiment

if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path,
                                         default_config_dir_path="configs/infer", # Will look for configuration files in this sub-directory
                                         default_exp_dir_path="/scratch/bailoni/projects/cellpose_inference_projects/jeany/2023-04-HTpaper" # Where to save the results (logs, checkpoints, segmentation results, etc.)
                                         )
    cls = CellposeBaseExperiment
    cls().run()

