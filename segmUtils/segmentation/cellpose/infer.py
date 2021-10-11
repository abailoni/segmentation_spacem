import os
from distutils.dir_util import copy_tree

import shutil

from segmfriends.utils.various import check_dir_and_create

from cellpose import models, io
import cv2
import numpy as np


def infer_cellpose_directory(in_dir, out_dir, keep_input_images_in_out_dir=False):
    # By default, CellPose outputs stuff in the same folder.
    # To avoid that, we copy images to the output folder and then delete them
    # FIXME: if the output is not empty, this makes a mess (read them as inputs, and delete them afterwards)
    assert not keep_input_images_in_out_dir, "Not implemented"
    check_dir_and_create(out_dir)
    shutil.rmtree(out_dir)
    check_dir_and_create(out_dir)
    copy_tree(in_dir, out_dir)

    input_files = []
    for root, dirs, files in os.walk(out_dir):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
                if not filename.startswith("."):
                    input_files.append([filename, root])



    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=True, model_type='cyto2')

    for filename, root in input_files:
        image = cv2.imread(os.path.join(root, filename))
        masks, flows, styles, diams = model.eval(image, diameter=None, channels=[2,0])

        _, file_extension = os.path.splitext(filename)

        # Plot result:
        import matplotlib.pyplot as plt
        from segmfriends.vis import plot_segm, get_figure, save_plot
        fig, ax = get_figure(1,1, figsize=(15,15))
        gray_img = image[...,1][None]
        plot_segm(ax, masks[None], background=gray_img, mask_value=0)
        save_plot(fig, root, filename.replace(file_extension, "_out_plot{}".format(file_extension)))
        plt.close(fig)

        cv2.imwrite(os.path.join(root, filename.replace(file_extension, "_segm{}".format(file_extension))),
                    masks.astype(np.uint16))


    # command = "python -m cellpose --dir {} --pretrained_model cyto2 --chan 2 --chan2 1 --use_gpu".format( # --no_npy --save_png
    #     out_dir
    # )
    # os.system(command)
    # stream = os.popen(command)
    # print(stream.read())



    # If needed, remove original image files:
    if not keep_input_images_in_out_dir:
        for filename, root in input_files:
            os.remove(os.path.join(root, filename))

    #




if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"
    input_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images/cellpose")
    out_dir = os.path.join(scratch_dir, "projects/spacem_segm/segm/cellpose_noDAPI")
    infer_cellpose_directory(input_dir, out_dir)




# import os
# os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/scratch/bailoni/.cellpose/models"
#
# from cellpose import models
# import skimage.io
#
#
# # model_type='cyto' or model_type='nuclei'
# model = models.Cellpose(gpu=True, model_type='cyto')
#

"""


"""
