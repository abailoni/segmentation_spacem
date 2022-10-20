import os

import numpy as np
from matplotlib import pyplot as plt

from segmUtils.preprocessing.preprocessing import read_uint8_img, read_segmentation_from_file
import  segmfriends.vis as vis_utils
import tifffile

dir = "/scratch/bailoni/projects/nastia/macrophages_segmentation/some_test_images"

for root, dirs, files in os.walk(dir):
    for filename in files:
        if filename.endswith(".tif"):
            img = tifffile.imread(os.path.join(root, filename))
            print(img.shape)
            # img = read_uint8_img(os.path.join(root, filename))

            # Average the three channels:
            dapi = np.copy(img[[2]])
            # img = img.mean(axis=0)[None]
            # img = (img[0] + img[2])[None]
            img = img[[0]]
            # print(img.shape)

            # Load segmentation:
            segm_path = os.path.join(root, "cellpose_predictions", filename.replace(".tif", "_cp_masks.png"))
            segm = read_segmentation_from_file(segm_path)[None]
            # print(segm.shape)

            out_folder = os.path.join(root, "overlayed_segmentations")
            os.makedirs(out_folder, exist_ok=True)

            fig, ax = vis_utils.get_figure(1,1,figsize=(10,10))
            vis_utils.plot_segm(ax, segm, background=img, mask_value=0)
            plt.tight_layout()
            vis_utils.save_plot(fig, out_folder, file_name=filename.replace(".tif", "_segmentation.png"))

            fig, ax = vis_utils.get_figure(1,1,figsize=(10,10))
            vis_utils.plot_segm(ax, np.zeros_like(segm), background=img, mask_value=0)
            plt.tight_layout()
            vis_utils.save_plot(fig, out_folder, file_name=filename.replace(".tif", ".png"))

            # fig, ax = vis_utils.get_figure(1,1,figsize=(10,10))
            # vis_utils.plot_segm(ax, np.zeros_like(segm), background=dapi, mask_value=0)
            # plt.tight_layout()
            # vis_utils.save_plot(fig, out_folder, file_name=filename.replace(".tif", "_dapi.png"))

