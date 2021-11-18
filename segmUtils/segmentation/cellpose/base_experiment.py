from copy import deepcopy
import os
import sys

import json
import numpy as np
from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmfriends.speedrun_exps.utils import process_speedrun_sys_argv

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

        # Replace some path placeholders:
        # TODO: move to another place and generalize to the full config?
        input_dir = self.get("cellpose_inference/input_dir")
        if input_dir is not None:
            if "$EXP_DIR" in input_dir:
                input_dir = input_dir.replace("$EXP_DIR", self.experiment_directory)
                self.set("cellpose_inference/input_dir", input_dir)

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
        data_zarr_group = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        datasets_to_process = self.get("preprocessing/datasets_to_process", ensure_exists=True)
        channels_to_process = self.get("preprocessing/channels_to_process", ensure_exists=True)
        kwargs_collected = []
        for dataset_name in datasets_to_process:
            dataset_kwargs = self.get("preprocessing/{}".format(dataset_name), ensure_exists=True)
            dataset_kwargs["dataset_name"] = dataset_name
            kwargs_collected.append(dataset_kwargs)

        # Get files from directories, find unique names, and create a unified stacked zarr file (for better visualization):
        nb_preprocessed_images = spacem_preproc.convert_multiple_dataset_to_zarr_group(data_zarr_group,
                                                                                       channels_to_process,
                                                                                       *kwargs_collected)
        self.set("nb_preprocessed_images", nb_preprocessed_images)

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
        self.convert_multiple_cellpose_output_to_zarr()

    def convert_multiple_cellpose_output_to_zarr(self):
        zarr_path_predictions, collected_model_names = convert_multiple_cellpose_output_to_zarr(
            os.path.join(self.experiment_directory, "cellpose_predictions"))

        # Save data in config file for later use:
        self.set("cellpose_inference/zarr_path_predictions", zarr_path_predictions)
        self.set("cellpose_inference/names_predicted_models", collected_model_names)

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
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path)

    cls = CellposeBaseExperiment
    cls().run()
