import os

import pandas
import numpy as np
import tifffile as tiff

# main_dir = "/g/scb/alexandr/shared/alberto"
main_dir = "/scratch/bailoni"

data_paths = {
    "Feeded_W3": {
        "post": "projects/gastrosome_processing/SpaceM_processing/Feeding_W3/input/microscopy/post-maldi/stitched_images/img_t1_z1_c0.tif",
        "pre": "projects/gastrosome_processing/SpaceM_processing/Feeding_W3/input/microscopy/pre-maldi/stitched_images/img_t1_z1_c0.tif"
    },
    # "Drug_W8": {
    #     "annotations": "projects/gastrosome_processing/annotations/Well3_marked/cropped_registered_pre_post_marked.csv",
    #     "input_image": "projects/gastrosome_processing/annotations/Well3_marked/cropped_registered_pre_post_marked.tif"
    # },
}


for data_name in data_paths:
    data_info = data_paths[data_name]
    post = tiff.imread(os.path.join(main_dir, data_info["post"]))
    print(post[:].max())
    print(post.shape)
    # pre = tiff.imread(os.path.join(main_dir, data_info["pre"]))
    # post[[0, -1],:] = 0
    # pre[[0, -1],:] = 0
    #
    #
    # tiff.imwrite(os.path.join(main_dir, data_info["post"]), post)
    # tiff.imwrite(os.path.join(main_dir, data_info["pre"]), pre)


