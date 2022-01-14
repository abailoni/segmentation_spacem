import glob
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

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, convert_multiple_cellpose_output_to_zarr


class CoreSpaceMExperiment(BaseExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(CoreSpaceMExperiment, self).__init__(experiment_directory)
        # Privates
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.auto_setup(update_git_revision=False)


    def replace_experiment_placeholder_in_config(self, name_config_property):
        config_property = self.get(name_config_property)
        if isinstance(config_property, dict):
            # In case of a dictionary, recurse:
            for sub_property in config_property:
                self.replace_experiment_placeholder_in_config("{}/{}".format(name_config_property,
                                                                             sub_property))
        else:
            if config_property is not None and isinstance(config_property, str):
                if "$EXP_DIR" in config_property:
                    config_property = config_property.replace("$EXP_DIR", self.experiment_directory)
                    self.set(name_config_property, config_property)




class CellposeBaseExperiment(CoreSpaceMExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(CellposeBaseExperiment, self).__init__(experiment_directory, config)

        # Replace some path placeholders:
        # TODO: move to another place and generalize to the full config?
        self.replace_experiment_placeholder_in_config("cellpose_inference")
        self.replace_experiment_placeholder_in_config("generate_cellpose_input")


        self._zarr_path_predictions = None

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

        # Get path where to output cellpose images:
        # Legacy option: `generate_for`
        generate_for = self.get("generate_cellpose_input/generate_for", None)
        all_input_dir = []
        if generate_for is not None:
            for method_name in generate_for:
                all_input_dir.append(self.get("{}/input_dir".format(method_name), ensure_exists=True))
        else:
            all_input_dir = [self.get("generate_cellpose_input/out_dir", ensure_exists=True)]

        for cellpose_input_dir in all_input_dir:
            # Convert images from zarr to cellpose format:
            spacem_preproc.from_zarr_to_cellpose(data_zarr_group, out_dir=cellpose_input_dir,
                                                 delete_previous=True,
                                                 **convert_to_cellpose_kwargs)

    def cellpose_inference(self):

        # Get run args and paths:
        cellpose_input_dir = self.get("cellpose_inference/input_dir", ensure_exists=True)
        cellpose_out_dir = os.path.join(self.experiment_directory, "cellpose_predictions", "$MODEL_NAME")
        dirs_to_process = [[cellpose_input_dir, cellpose_out_dir]]

        multiple_cellpose_inference_kwargs = self.get("cellpose_inference/multiple_cellpose_inference_kwargs",
                                                      ensure_exists=True)

        # If necessary, replace the model path and get it from a given experiment:
        tested_models = multiple_cellpose_inference_kwargs["tested_models"]
        for model_name in tested_models:
            model_path = tested_models[model_name]
            if "$EXP_DIR" in model_path:
                model_path = model_path.replace("$EXP_DIR", self.experiment_directory)
            if model_path.endswith("$LATEST"):
                model_path = model_path.replace("$LATEST", "")
                # If experiment directory is not specified, append it:
                if not os.path.exists(model_path):
                    model_path = os.path.join(self.experiment_directory, "..", model_path)
                model_path = os.path.join(model_path, "Weights/models")
                assert os.path.exists(model_path), "Model path {} was not found".format(model_path)

                # Now find the most recent model in the specified experiment folder:
                list_of_files = glob.glob(os.path.join(model_path, "*"))  # * means all if need specific format then *.csv
                latest_model = max(list_of_files, key=os.path.getctime)
                model_path = latest_model

            # Save the modified path:
            tested_models[model_name] = model_path

        cellpose_infer.multiple_cellpose_inference(dirs_to_process=dirs_to_process,
                                                   **multiple_cellpose_inference_kwargs)

        # # Collect predictions from all models and combine all images in a single zarr file:
        # self.convert_multiple_cellpose_output_to_zarr()

    def convert_multiple_cellpose_output_to_zarr(self):
        data_zarr_group = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        csv_file = data_zarr_group.replace(".zarr", ".csv")
        print("Converting cellpose results to zarr")
        zarr_path_predictions, collected_model_names = convert_multiple_cellpose_output_to_zarr(
            os.path.join(self.experiment_directory, "cellpose_predictions"),
        csv_file=csv_file)

        # Save data in config file:
        self.set("cellpose_inference/zarr_path_predictions", self.zarr_path_predictions)
        self.set("cellpose_inference/names_predicted_models", collected_model_names)

    def export_results(self):
        """
        For the moment this method is thought for inference. Generalize...?
        """
        zarr_path_predictions = self.zarr_path_predictions
        input_zarr_group_path = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        export_images_from_zarr_kwargs = self.get("export_results/export_images_from_zarr_kwargs", ensure_exists=True)
        csv_config_path = input_zarr_group_path.replace(".zarr", ".csv")

        export_dir = os.path.join(self.experiment_directory, "exported_results")
        self.set("export_results/export_path", export_dir)

        # Insert zarr path of the prediction file in the export parameters:
        assert "datasets_to_export" in export_images_from_zarr_kwargs
        datasets_to_export = export_images_from_zarr_kwargs.pop("datasets_to_export")
        for idx in range(len(datasets_to_export)):
            datasets_to_export[idx]["z_path"] = zarr_path_predictions

        print("Exporting result images to original folder structure...")

        # Export images in the original structure:
        export_images_from_zarr(export_dir,
                                csv_config_path,
                                datasets_to_export=datasets_to_export,
                                delete_previous=True,
                                **export_images_from_zarr_kwargs)

    def compute_scores(self):
        configs_to_evaluate = self.get("compute_scores/configs_to_evaluate", ensure_exists=True)
        export_path = self.get("export_results/export_path", ensure_exists=True)

        collected_scores_names = None
        collected_scores = []

        global_config = self.get("compute_scores/global_config", {})

        for idx, export_name in enumerate(configs_to_evaluate):
            export_kwargs = deepcopy(global_config)
            export_kwargs.update(self.get("compute_scores/{}".format(export_name), {}))
            dataset_name = export_kwargs.pop("dataset_name")
            export_kwargs["pred_extension"] = ".tif"
            export_kwargs["pred_dir"] = pred_dir = os.path.join(export_path, dataset_name)
            AP_thresholds = export_kwargs["AP_thresholds"]

            scores = compute_scores(**export_kwargs)

            # Prepare scores to be written to csv file:
            scores_names = []
            new_collected_scores = []
            for sc_name in scores:
                score = scores[sc_name]
                assert len(score.shape) == 1
                for AP_thr_indx, scr in enumerate(score):
                    new_collected_scores.append(scr)
                    if collected_scores_names is None:
                        if score.shape[0] == 1:
                            scores_names.append(sc_name)
                        else:
                            assert score.shape[0] == len(AP_thresholds)
                            scores_names.append("{}_{}".format(sc_name, AP_thresholds[AP_thr_indx]))
            if collected_scores_names is None:
                collected_scores_names = deepcopy(scores_names)
            basedir = os.path.basename(os.path.normpath(pred_dir))

            estimate_diam = None
            # if "_noDiamEst" in model_name:
            #     estimate_diam = 0
            # elif "_diamEst" in model_name:
            #     estimate_diam = 1
            # else:
            #     estimate_diam = None

            collected_scores.append([basedir, export_name, estimate_diam] + new_collected_scores)
            print("Done {}, {}".format(export_name, pred_dir))

        # Create output score directory:
        out_score_dir = os.path.join(self.experiment_directory, "scores")
        check_dir_and_create(out_score_dir)
        df = pandas.DataFrame(collected_scores,
                          columns=['Data type', 'Model name', 'Estimated cell size'] + collected_scores_names)
        df.sort_values(by=['Data type', 'aji'], inplace=True, ascending=False)
        df.to_csv(os.path.join(out_score_dir, "scores.csv"))

    @property
    def zarr_path_predictions(self):
        if self._zarr_path_predictions is None:
            self._zarr_path_predictions = os.path.join(self.experiment_directory, "cellpose_predictions",
                                                    "predictions_collected.zarr")
        return self._zarr_path_predictions



if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path)

    cls = CellposeBaseExperiment
    cls().run()
