import os.path
import sys

# import numpy as np
# import vigra
# import segmfriends.io.zarr as zarr_utils
# import nifty.graph.rag
# import nifty.tools as ntools

# from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmfriends.speedrun_exps.utils import process_speedrun_sys_argv

# from speedrun import BaseExperiment
# from segmfriends.utils.paths import get_vars_from_argv_and_pop

from segmUtils.segmentation.cellpose.base_experiment import CellposeBaseExperiment, CoreSpaceMExperiment

import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

# from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
# import json, cv2, random
import imageio
# import shutil
from copy import deepcopy
# import pandas as pd
# import vigra

from segmfriends.utils.various import check_dir_and_create, yaml2dict
import segmfriends.vis as segm_vis

import matplotlib.pyplot as plt
import matplotlib
# import numpy as np
from segmUtils.preprocessing.preprocessing import read_uint8_img

matplotlib.rc('font',family='sans-serif', size=8)
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

class MakeComparisonPlots(CoreSpaceMExperiment):
    def run(self):
        self.make_comparison_plots()

        # In case some parameter was updated, dump configuration again:
        self.dump_configuration()

    def make_comparison_plots(self):
        plot_flows = self.get("plot_config/plot_flows")
        assert not plot_flows

        PLOT_DIR = os.path.join(self.experiment_directory, "Plots")
        check_dir_and_create(PLOT_DIR)

        fig_size = 10

        configs_to_plot = self.get("experiment_config/configs_to_plot")
        global_config = self.get("experiment_config/global_config", {})

        # TODO: generalize to multiple datasets
        # Get all database names:
        dataset_names = self.get("experiment_config/dataset_names")

        # Get configs:
        all_configs = []
        for config_name in configs_to_plot:
            config = deepcopy(global_config)
            config.update(self.get("experiment_config/{}".format(config_name), {}))
            all_configs.append(config)

        for dataset_name in dataset_names:
            dataset_plot_dir = os.path.join(PLOT_DIR, dataset_name)
            print("Exporting plots to ", dataset_plot_dir)
            check_dir_and_create(dataset_plot_dir)

            # Get prediction directories paths:
            all_pred_dirs = []
            for i, config_name in enumerate(configs_to_plot):
                all_pred_dirs.append(os.path.join(self.get("experiment_config/main_experiment_dir"),
                                        all_configs[i]["exp_name"], "exported_results", dataset_name))

            nb_models = len(all_configs)
            # Temporary get first config:
            first_config = all_configs[0]
            first_pred_path = all_pred_dirs[0]

            GT_config = self.get("GT_config/{}".format(dataset_name), None)
            raw_config = self.get("raw_config/{}".format(dataset_name))

            # Loop over images of the first prediction directory:
            for root, dirs, files in os.walk(first_pred_path):
                for filename in files:
                    rel_path = os.path.relpath(root, first_pred_path)
                    pred_basename, pred_extension = os.path.splitext(filename)

                    # Check extension and filter prediction files:
                    pred_filter = first_config.get("pred_filter", None)
                    pred_filter = None if pred_filter == "" else pred_filter
                    if pred_filter is not None:
                        if not pred_basename.endswith(first_config["pred_filter"]):
                            continue
                    if first_config.get("pred_extension", None) is not None:
                        if pred_extension != first_config["pred_extension"]:
                            continue


                    # ax_offset = 2 if GT_DIR[dataset_name] is not None else 1
                    ax_offset = 1

                    plt.subplots_adjust(
                        wspace=0.,
                        hspace=0.)

                    # nb_rows = 2 if plot_flows else 1
                    nb_rows = 2

                    f, ax = plt.subplots(ncols=(nb_models + ax_offset), nrows=nb_rows,
                                         figsize=(fig_size * (nb_models + ax_offset - 1), fig_size))
                    if nb_rows == 1:
                        ax = np.array([ax])
                    # f, ax = plt.subplots(ncols=1, nrows=(nb_models+ax_offset),
                    #                      figsize=(10, 10*(nb_models+ax_offset-1)))
                    for a in f.get_axes():
                        a.axis('off')

                    # -------------------------------
                    # Import raw image and GT:
                    # -------------------------------
                    GT_img_name = None
                    if pred_filter is None:
                        npy_filename = "{}_seg.npy".format(pred_basename)
                        raw_img_name = "{}{}{}".format(pred_basename, raw_config["filter"],
                                                       raw_config["extension"])
                        out_name = "{}_compare_plot.png".format(pred_basename)
                        if GT_config is not None:
                            GT_img_name = "{}{}{}".format(pred_basename, GT_config["filter"],
                                                       GT_config["extension"])
                    else:
                        npy_filename = pred_basename.replace(pred_filter, "_seg.npy")
                        out_name = pred_basename.replace(pred_filter, "_compare_plot.png")
                        raw_img_name = pred_basename.replace(pred_filter, raw_config["filter"]) + raw_config["extension"]
                        if GT_config is not None:
                            GT_img_name = pred_basename.replace(pred_filter, GT_config["filter"]) + GT_config["extension"]

                    # Load raw:
                    raw = read_uint8_img(os.path.join(raw_config["dir"], rel_path, raw_img_name),
                                         add_all_channels_if_needed=True)
                    raw = raw[:, :, 0]  # Only keep one channel
                    ax[0, 0].matshow(raw, cmap="gray")
                    ax[0, 0].set_title("Input image", fontweight='bold')

                    # Load GT if needed:
                    if GT_config is not None:
                        gt_segm = imageio.imread(os.path.join(GT_config["dir"], rel_path, GT_img_name))
                        segm_vis.plot_segm(ax[1, 0], gt_segm[None], background=raw[None], alpha_labels=0.45,
                                           alpha_boundary=0.4, mask_value=0)
                        ax[1, 0].set_title("Ground truth", fontweight='bold')


                    # -------------------------------
                    # Plot predictions:
                    # -------------------------------
                    for idx, config in enumerate(all_configs):
                        model_name_to_plot = configs_to_plot[idx]
                        pred_dir = all_pred_dirs[idx]
                        # Load segm:
                        new_pred_path = os.path.join(pred_dir, rel_path, filename)
                        assert os.path.isfile(new_pred_path), "Prediction {} not found for model {}: {}".format(
                            filename,
                            model_name_to_plot,
                            new_pred_path)
                        segm = imageio.imread(new_pred_path)
                        # prediction_data = np.load(os.path.join(PREDICTION_DIR, model_name, dataset_name, npy_filenam"), +.item""
                        # flows = prediction_data['flows'][0][0]

                        # ax[1, model_idx + ax_offset].imshow(flows)

                        segm_vis.plot_segm(ax[0, idx + ax_offset], segm[None], background=raw[None],
                                           alpha_labels=0.45, alpha_boundary=0.4, mask_value=0)
                        ax[0, idx + ax_offset].set_title(model_name_to_plot, fontweight='bold')

                    plt.subplots_adjust(
                        wspace=0.,
                        hspace=0.)

                    f.tight_layout()
                    check_dir_and_create(os.path.join(dataset_plot_dir, rel_path))
                    f.savefig(os.path.join(dataset_plot_dir, rel_path, out_name),
                              format='png', bbox_inches='tight')
                    print("Plotted {}".format(out_name))



if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path, default_config_rel_path="./configs",
                                         default_exp_path="/scratch/bailoni/projects/cellpose_projects/combined_plots")

    cls = MakeComparisonPlots
    cls().run()
