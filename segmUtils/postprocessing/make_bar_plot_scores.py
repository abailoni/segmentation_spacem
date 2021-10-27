import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import imageio
import shutil
from copy import deepcopy
import pandas as pd
import vigra

from segmfriends.utils.various import check_dir_and_create

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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

if __name__ == "__main__":


    PLOT_DIR = "/scratch/bailoni/projects/train_cellpose/plots"
    # df = pd.read_csv("/scratch/bailoni/projects/train_cellpose/scores_with_estimated_diameters_all.csv")
    df = pd.read_csv("/scratch/bailoni/projects/train_cellpose/scores_cleaned_LIVECell.csv")

    # df.loc[:, "aji"].to_numpy()

    # models_to_plot = ["cyto2_diamEst", "trained_on_LIVECell_diamEst", "trained_on_cellpose_diamEst", "cyto_diamEst",
    #                   "finetuned_LIVECell_lr_02_diamEst"]

    # # ALL MODELS:
    # models_to_plot = {"cyto2_diamEst": "CellPose cyto2 ",
    #                   "cyto_diamEst": "CellPose cyto",
    #                   "trained_on_LIVECell_noDiamEst": "Trained on LIVECell+CellPose data",
    #                   "trained_on_LIVECell_diamEst": "Trained on LIVECell+CellPose data (est diam)",
    #                   "trained_on_cellpose_noDiamEst": "Trained on CellPose data",
    #                   "trained_on_cellpose_diamEst": "Trained on CellPose data (est diam)",
    #                   "finetuned_LIVECell_lr_02_noDiamEst": "finetuned_LIVECell_lr_02_noDiamEst", # "Fine-tuned on LIVECell+CellPose data"
    #                   "finetuned_LIVECell_lr_00002_noDiamEst": "finetuned_LIVECell_lr_00002_noDiamEst",
    #                   # "finetuned_LIVECell_lr_00002_diamEst": "finetuned_LIVECell_lr_00002_diamEst",
    #                   "finetuned_LIVECell_lr_02_diamEst": "finetuned_LIVECell_lr_02_diamEst",
    #                   }
    # spacing = 4


    models_to_plot = {"cyto2_diamEst": "CellPose cyto2 model",
                      # "cyto_diamEst": "CellPose cyto",
                      # "trained_on_LIVECell_noDiamEst": "Trained on LIVECell+CellPose from scratch",
                      # "trained_on_LIVECell_diamEst": "Trained on LIVECell+CellPose data (est diam)",
                      # "trained_on_cellpose_noDiamEst": "Trained on CellPose data",
                      # "trained_on_cellpose_diamEst": "Trained on CellPose data (est diam)",
                      # "finetuned_LIVECell_lr_02_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_noDiamEst": "cyto2 finetuned on LIVECell",
                      "cleaned_finetuned_LIVECell_v1_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "cleaned_from_scratch_LIVECell_v1_noDiamEst": "cleaned_from_scratch_LIVECell_v1_noDiamEst",
                      # "finetuned_LIVECell_lr_00002_diamEst": "finetuned_LIVECell_lr_00002_diamEst",
                      # "finetuned_LIVECell_lr_02_diamEst": "finetuned_LIVECell_lr_02_diamEst",
                      }
    spacing = 2


    # scores_to_plot = ["aji", "fp_0.5", "fn_0.5", "ap_0.5", "tp_0.5"]
    # scores_to_plot_names = ["Aggregated Jaccard Index", "False positives 0.5", "False negatives 0.5",
    #                         "Average precision 0.5", "True positives 0.5"]

    scores_to_plot = ["aji", "fp_0.9", "fn_0.9", "ap_0.9", "tp_0.9"]
    scores_to_plot_names = ["Aggregated Jaccard Index", "False positives 0.9", "False negatives 0.9",
                            "Average precision 0.9", "True positives 0.9"]

    labels = {
        "LIVECell_test_cleaned": 'LIVECell test data',
        # "LIVECell_test": 'LIVECell test data',
        "cellpose_test":'CellPose test data',
        "alex": 'Alex images'}
    labels_to_plot = [lb for _, lb in labels.items()]

    bar_width = 0.25  # the width of the bars





    # ax = df.plot.bar(rot=0)

    for score_idx, score_name in enumerate(scores_to_plot):
        r = np.arange(0, len(labels) * spacing, spacing)
        fig, ax = plt.subplots()

        # sub_df = df[["Model name", "Data type", score_name]]
        # ax = df.plot.bar(rot=0)


        nb_models = len(models_to_plot)
        for mdoel_idx, model_name in enumerate(models_to_plot):
            model_name_to_plot = models_to_plot[model_name] # model_name.replace("_diamEst", "").replace("_noDiamEst", "")
            model_results = df.loc[df["Model name"] == model_name]

            model_scores = []
            for data_label in labels:
                score = model_results.loc[model_results["Data type"] == data_label, score_name]
                assert len(score) ==  1
                model_scores.append(float(score.to_numpy()[0]))
            rects = ax.bar(r, model_scores, bar_width, edgecolor='white', label=model_name_to_plot, )
            # plt.hlines(y=r, xmin=0, xmax=df['percentage'], color='#007acc', alpha=0.2, linewidth=5)
            r = [x + bar_width for x in r]
            # ax.bar_label(rects, padding=3)
        ax.set_ylabel(scores_to_plot_names[score_idx])
        ax.set_title('Comparison between different CellPose models')
        if score_idx in [0, 3] and "0.5" in score_name:
            ax.set_ylim([0,1.])
        # print([x + (nb_models/2.)*bar_width for x in range(0, len(labels) * spacing, spacing)])
        # print([x +  bar_width for x in range(0, len(labels) * spacing, spacing)])
        ax.set_xticks([x + (nb_models/2.-0.5)*bar_width for x in range(0, len(labels) * spacing, spacing)])
        ax.set_xticklabels(labels_to_plot)
        # ax.set_xlabel('Datasets', fontweight='bold')
        ax.legend()

        # fig.tight_layout()

        fig.savefig(os.path.join(PLOT_DIR, "plot_{}.pdf".format(score_name)), format='pdf')


