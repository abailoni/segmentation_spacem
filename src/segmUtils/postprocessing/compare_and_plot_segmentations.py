import numpy as np
# from cellpose.metrics import aggregated_jaccard_index, average_precision

import os
import json, cv2, random
import imageio
import shutil
from copy import deepcopy

from segmfriends.utils.various import check_dir_and_create
import segmfriends.vis as segm_vis

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
    # -----------------------
    # Parameters to choose:
    # -----------------------
    pred_imgs_to_plot = {
        # "alex": [
        #         "0_valid_s1w3_img_t1_z1_c0_0_img_cp_masks.png",
        #         "4_train_s1w3_img_t1_z1_c0_0_img_cp_masks.png",
        #     "7_train_s2w3_img_t1_z1_c0_0_img_cp_masks.png"
        #         ],
        "cropped_images": [
            "0_train_s1w3_img_t1_z1_c0_0_img_cp_masks.png",
            "1_train_s4w3_img_t1_z1_c0_0_img_cp_masks.png",
            "2_train_s3w3_img_t1_z1_c0_0_img_cp_masks.png",
            "3_train_s2w3_img_t1_z1_c0_0_img_cp_masks.png",
            # "4_W4_20210823_AB_DKFZCocultur_analysis_img_t1_z1_c0_0_img_cp_masks.png",
            # "5_W4_20210823_AB_DKFZCocultur_analysis_img_t1_z1_c0_1_img_cp_masks.png",
            # "6_W4_20210823_AB_DKFZCocultur_analysis_img_t1_z1_c0_2_img_cp_masks.png",
            # "7_W4_20210823_AB_DKFZCocultur_analysis_img_t1_z1_c0_3_img_cp_masks.png",
            # "59_cultured_HeLa_img_t1_z1_c0_1_img_cp_masks.png",
            # "58_cultured_HeLa_img_t1_z1_c0_0_img_cp_masks.png",
            # "106_cytospinned_HeLa_img_t1_z1_c0_0_img_cp_masks.png",
            # "107_cytospinned_HeLa_img_t1_z1_c0_1_img_cp_masks.png",
            # "8_cultured_keratinocytes_img_t1_z1_c0_0_img_cp_masks.png",
            # "9_cultured_keratinocytes_img_t1_z1_c0_1_img_cp_masks.png",
        ],
        # "LIVECell_test": [
        #     "A172_Phase_C7_1_00d04h00m_4_cp_masks.png",
        #     "A172_Phase_C7_1_02d16h00m_3_cp_masks.png",
        #     "BT474_Phase_D3_2_04d12h00m_1_cp_masks.png",
        #     "BV2_Phase_A4_2_03d00h00m_1_cp_masks.png",
        #     "Huh7_Phase_A12_2_04d00h00m_3_cp_masks.png",
        #     "MCF7_Phase_H4_1_01d12h00m_2_cp_masks.png",
        #     "SHSY5Y_Phase_A10_1_00d16h00m_4_cp_masks.png",
        #     "SkBr3_Phase_G3_1_03d12h00m_2_cp_masks.png"
        # ]
    }


    models_to_compare = {
                        "cyto2_diamEst": "cyto2 model",
                      # "cyto_diamEst": "CellPose cyto",
                      # "trained_on_LIVECell_noDiamEst": "Trained on LIVECell+CellPose from scratch",
                      # "trained_on_LIVECell_diamEst": "Trained on LIVECell+CellPose data (est diam)",
                      # "trained_on_cellpose_noDiamEst": "Trained on CellPose data",
                      # "trained_on_cellpose_diamEst": "Trained on CellPose data (est diam)",
                      # "finetuned_LIVECell_lr_02_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_noDiamEst": "cyto2 finetuned on LIVECell+CellPose",
                      # "finetuned_LIVECell_lr_00002_diamEst": "finetuned_LIVECell_lr_00002_diamEst",
                      # "finetuned_LIVECell_lr_02_diamEst": "finetuned_LIVECell_lr_02_diamEst",
                      "cleaned_finetuned_LIVECell_v1_noDiamEst": "Finetuned on LIVECell",
                      # "cleaned_from_scratch_LIVECell_v1_noDiamEst": "cleaned_from_scratch_LIVECell_v1_noDiamEst",
                      # "cleaned_LIVECell_pip_noDiamEst": "cleaned_LIVECell_pip",
                      #   "full_LIVECell_lr_002_SGD_cyto2_noDiamEst": "full_LIVECell_lr_002_SGD_cyto2",
                      }
    # -----------------------



    RAW_IMAGE_DIR = {
        "LIVECell_test": "/scratch/bailoni/projects/train_cellpose/data/test",
        "cellpose_test": "/scratch/bailoni/projects/train_cellpose/data/test",
        "alex": "/scratch/bailoni/projects/spacem_segm/alex_labeled/cellpose",
        "cropped_images": "/scratch/bailoni/projects/spacem_segm/input_images_small/cellpose",
    }
    GT_DIR = {
        "LIVECell_test": "/scratch/bailoni/projects/train_cellpose/data/test",
        "cellpose_test": "/scratch/bailoni/projects/train_cellpose/data/test",
        "alex": "/scratch/bailoni/datasets/alex/labels",
        "cropped_images": None
    }

    GT_filter = "_masks"
    pred_filter = "_cp_masks"
    raw_filter = ""
    raw_extension = ".png"

    PREDICTION_DIR = "/scratch/bailoni/projects/train_cellpose/predictions"

    PLOT_DIR = "/scratch/bailoni/projects/train_cellpose/compare_segm_plots"
    check_dir_and_create(PLOT_DIR)


    # labels = {
    #     "LIVECell_test": 'LIVECell test data',
    #     "cellpose_test":'CellPose test data',
    #     "alex": 'Alex images'}
    # labels_to_plot = [lb for _, lb in labels.items()]
    #


    fig_size = 10

    for dataset_name in pred_imgs_to_plot:
        # ax_offset = 2 if GT_DIR[dataset_name] is not None else 1
        ax_offset = 1
        for pred_img_name in pred_imgs_to_plot[dataset_name]:
            print(pred_img_name)
            pred_basename, pred_extension = os.path.splitext(pred_img_name)
            nb_models = len(models_to_compare)

            plt.subplots_adjust(
                            wspace=0.,
                            hspace=0.)


            f, ax = plt.subplots(ncols=(nb_models+ax_offset), nrows=2,
                                 figsize=(fig_size*(nb_models+ax_offset-1), fig_size))
            # f, ax = plt.subplots(ncols=1, nrows=(nb_models+ax_offset),
            #                      figsize=(10, 10*(nb_models+ax_offset-1)))
            for a in f.get_axes():
                a.axis('off')

            npy_filename = pred_basename.replace(pred_filter, '_seg.npy')

            # FIXME: replacing filter in raw and GT has problems if pred_filter == ""
            # Load raw and GT:
            raw_img_name = pred_img_name.replace(pred_filter, raw_filter).replace(pred_extension, raw_extension)
            raw = cv2.imread(os.path.join(RAW_IMAGE_DIR[dataset_name], raw_img_name))
            raw = raw[:,:,1] # Only keep green channel
            ax[0,0].matshow(raw, cmap="gray")
            ax[0,0].set_title("Input image", fontweight='bold')

            if GT_DIR[dataset_name] is not None:
                gt_name = pred_img_name.replace(pred_filter, GT_filter)
                gt_segm = imageio.imread(os.path.join(GT_DIR[dataset_name], gt_name))
                segm_vis.plot_segm(ax[1,0], gt_segm[None], background=raw[None], alpha_labels=0.45, alpha_boundary=0.4, mask_value=0)
                ax[1,0].set_title("Ground truth", fontweight='bold')

            for model_idx, model_name in enumerate(models_to_compare):
                model_name_to_plot = models_to_compare[model_name]

                prediction_data = np.load(os.path.join(PREDICTION_DIR, model_name, dataset_name,npy_filename), allow_pickle=True).item()

                # prediction_data['img']
                # prediction_data['masks']
                flows = prediction_data['flows'][0][0]
                # Load segm
                segm = imageio.imread(os.path.join(PREDICTION_DIR, model_name, dataset_name, pred_img_name))


                ax[1,model_idx+ax_offset].imshow(flows)

                segm_vis.plot_segm(ax[0, model_idx+ax_offset], segm[None], background=raw[None], alpha_labels=0.45, alpha_boundary=0.4, mask_value=0)
                ax[0, model_idx+ax_offset].set_title(model_name_to_plot, fontweight='bold')

            # f.suptitle('Predictions of different CellPose models', fontweight='bold')

            plt.subplots_adjust(
                            wspace=0.,
                            hspace=0.)


            f.tight_layout()

            check_dir_and_create(os.path.join(PLOT_DIR, dataset_name))
            f.savefig(os.path.join(PLOT_DIR, dataset_name, "{}_compare_plot.png".format(pred_basename.replace(pred_filter, ""))),
                      format='png', bbox_inches='tight')




