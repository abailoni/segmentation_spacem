import os.path
import sys

from segmUtils.segmentation.cellpose.train_experiment import TrainingCellposeExperiment
from speedrun import process_speedrun_sys_argv

if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path, default_config_dir_path="./configs/train",
                                         default_exp_dir_path="/scratch/bailoni/projects/cellpose_projects/finetuning/train")

    cls = TrainingCellposeExperiment
    cls().run()

