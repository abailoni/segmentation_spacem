import os

import numpy as np
from matplotlib import pyplot as plt

from segmUtils.preprocessing.preprocessing import read_uint8_img, read_segmentation_from_file
import  segmfriends.vis as vis_utils
import tifffile

dir = "/scratch/bailoni/projects/spacem_cellpose_inference/jeany/oct_22/debug/cellpose_inputs"
# extension_input = ".tif"
extension_input = ".png"

predictions_dir = "/scratch/bailoni/projects/spacem_cellpose_inference/jeany/oct_22/debug/cellpose_predictions/trained_model_noDiamEst/"
out_folder = "/scratch/bailoni/projects/spacem_cellpose_inference/jeany/oct_22/debug/overlayed_predictions"
FIG_SIZE = 25


for root, dirs, files in os.walk(dir):
    for filename in files:
        if filename.endswith(extension_input):
            print(filename)
            if extension_input == ".png":
                img = read_uint8_img(os.path.join(root, filename))
                img =  np.rollaxis(img, axis=2, start=0)
            elif extension_input == ".tif":
                img = tifffile.imread(os.path.join(root, filename))
            else:
                raise ValueError(extension_input)

            print(img.shape)



            img = img[[1]]
            # # Average the three channels:
            # # TRANS = np.copy(img[[2]])
            # # img = img.mean(axis=0)[None]
            # # img = (img[0] + img[2])[None]
            # # print(img.shape)

            # Load segmentation:
            segm_path = os.path.join(predictions_dir, filename.replace(extension_input, "_cp_masks.png"))
            segm = read_segmentation_from_file(segm_path)[None]
            # print(segm.shape)

            os.makedirs(out_folder, exist_ok=True)

            fig, ax = vis_utils.get_figure(1,1,figsize=(FIG_SIZE,FIG_SIZE))
            vis_utils.plot_segm(ax, segm, background=img, mask_value=0)
            plt.tight_layout()
            vis_utils.save_plot(fig, out_folder, file_name=filename.replace(extension_input, "_segmentation.png"))

            fig, ax = vis_utils.get_figure(1,1,figsize=(FIG_SIZE,FIG_SIZE))
            vis_utils.plot_segm(ax, np.zeros_like(segm), background=img, mask_value=0)
            plt.tight_layout()
            vis_utils.save_plot(fig, out_folder, file_name=filename.replace(extension_input, ".png"))

            # fig, ax = vis_utils.get_figure(1,1,figsize=(FIG_SIZE,FIG_SIZE))
            # vis_utils.plot_segm(ax, np.zeros_like(segm), background=dapi, mask_value=0)
            # plt.tight_layout()
            # vis_utils.save_plot(fig, out_folder, file_name=filename.replace(".tif", "_dapi.png"))
            plt.close('all')

