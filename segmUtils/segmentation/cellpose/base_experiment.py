from copy import deepcopy
import os
import sys

import json
import numpy as np
from segmUtils.io.export_images_from_zarr import export_images_from_zarr

from speedrun import BaseExperiment
from segmfriends.utils.paths import get_vars_from_argv_and_pop

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, convert_multiple_cellpose_output_to_zarr


class CellposeBaseExperiment(BaseExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(CellposeBaseExperiment, self).__init__(experiment_directory)
        # Privates
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.auto_setup(update_git_revision=False)

    def run(self):
        methods_to_run = self.get("methods_to_run", ensure_exists=True)
        assert isinstance(methods_to_run, list)

        for method_name in methods_to_run:
            method = getattr(self, method_name, None)
            assert method is not None, "Method {} is not defined in the experiment!".format(method_name)
            method()

        # In case some parameter was updated, dump configuration again:
        self.dump_configuration()

    def preprocessing(self):
        # Get run args and paths:
        in_dir = self.get("input_paths/main_data_dir", ensure_exists=True)
        data_zarr_group = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        convert_images_to_zarr_dataset_kwargs = self.get("preprocessing/convert_images_to_zarr_dataset_kwargs",
                                                         ensure_exists=True)

        # Get files from directories, find unique names, and create a unified stacked zarr file (for better visualization):
        # TODO: generalize to multiple imports from multiple folders
        idx_images = spacem_preproc.convert_images_to_zarr_dataset(in_dir,
                                                                   out_zarr_path=data_zarr_group,
                                                                   **convert_images_to_zarr_dataset_kwargs)

    def generate_cellpose_input(self):
        # Get run args and paths:
        data_zarr_group = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        convert_to_cellpose_kwargs = self.get("generate_cellpose_input/convert_to_cellpose_kwargs", ensure_exists=True)

        generate_for = self.get("generate_cellpose_input/generate_for", ensure_exists=True)
        for method_name in generate_for:
            cellpose_input_dir = self.get("{}/input_dir".format(method_name), ensure_exists=True)

            # Convert images from zarr to cellpose format:
            spacem_preproc.from_zarr_to_cellpose(data_zarr_group, out_dir=cellpose_input_dir,
                                                 **convert_to_cellpose_kwargs)

    def cellpose_inference(self):
        # Get run args and paths:
        cellpose_input_dir = self.get("cellpose_inference/input_dir", ensure_exists=True)
        cellpose_out_dir = os.path.join(self.experiment_directory, "cellpose_predictions", "$MODEL_NAME")
        dirs_to_process = [[cellpose_input_dir, cellpose_out_dir]]

        multiple_cellpose_inference_kwargs = self.get("cellpose_inference/multiple_cellpose_inference_kwargs",
                                                      ensure_exists=True)
        cellpose_infer.multiple_cellpose_inference(dirs_to_process=dirs_to_process,
                                                   **multiple_cellpose_inference_kwargs)

        # Collect predictions from all models and combine all images in a single zarr file:
        zarr_path_predictions = convert_multiple_cellpose_output_to_zarr(
            os.path.join(self.experiment_directory, "cellpose_predictions"))
        self.set("cellpose_inference/zarr_path_predictions", zarr_path_predictions)

    def export_results(self):
        """
        For the moment this method is thought for inference. Generalize...?
        """
        zarr_path_predictions = self.get("cellpose_inference/zarr_path_predictions", ensure_exists=True)
        input_zarr_group_path = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        export_images_from_zarr_kwargs = self.get("export_results/export_images_from_zarr_kwargs", ensure_exists=True)
        csv_config_path = input_zarr_group_path.replace(".zarr", ".csv")

        export_dir = os.path.join(self.experiment_directory, "exported_results")

        # Insert zarr path of the prediction file in the export parameters:
        assert "datasets_to_export" in export_images_from_zarr_kwargs
        datasets_to_export = export_images_from_zarr_kwargs.pop("datasets_to_export")
        for idx in range(len(datasets_to_export)):
            datasets_to_export[idx]["z_path"] = zarr_path_predictions

        # Export images in the original structure:
        export_images_from_zarr(export_dir,
                                csv_config_path,
                                datasets_to_export=datasets_to_export,
                                **export_images_from_zarr_kwargs)

    # ------------------------------------------------------------------------------------------------------
    # Basic modifications to change the name of the configuration file from 'train_config' to 'main_config'
    def read_config_file(self, file_name='main_config.yml', **kwargs):
        return super(CellposeBaseExperiment, self).read_config_file(file_name=file_name, **kwargs)

    def inherit_configuration(self, from_experiment_directory, file_name='main_config.yml', **kwargs):
        return super(CellposeBaseExperiment, self).inherit_configuration(from_experiment_directory, file_name=file_name,
                                                                         **kwargs)

    def dump_configuration(self, file_name='main_config.yml'):
        return super(CellposeBaseExperiment, self).dump_configuration(file_name=file_name)
    # ------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))

    collected_paths, sys.argv = get_vars_from_argv_and_pop(sys.argv,
                                                           config_path=os.path.join(source_path, '../../../configs'),
                                                           exp_path="/scratch/bailoni/projects/cellpose_projects")
    config_path = collected_paths["config_path"]
    experiments_path = collected_paths["exp_path"]

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = os.path.join(config_path, sys.argv[i])
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[ind])
            i += 1
        else:
            break
    cls = CellposeBaseExperiment
    cls().run()
