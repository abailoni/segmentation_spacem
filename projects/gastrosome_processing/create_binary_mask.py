import os

import pandas
import numpy as np
import tifffile as tiff

# main_dir = "/g/scb/alexandr/shared/alberto"
main_dir = "/scratch/bailoni"

data_paths = {
    "Drug_W8": {
        "annotations": "projects/gastrosome_processing/annotations/Drug_well8/DrugW8_registered_crop_marked.csv",
        "input_image": "projects/gastrosome_processing/annotations/Drug_well8/DrugW8_registered_crop_marked.tif"
    },
    "Feeded_W3": {
        "annotations": "projects/gastrosome_processing/annotations/Well3_marked/cropped_registered_pre_post_marked.csv",
        "input_image": "projects/gastrosome_processing/annotations/Well3_marked/cropped_registered_pre_post_marked.tif"
    },
}



for data_name in data_paths:
    data_info = data_paths[data_name]
    image = tiff.imread(os.path.join(main_dir, data_info["input_image"]))
    binary_mask = np.zeros(shape=(1,) + image.shape[1:], dtype="uint16")

    df = pandas.read_csv(os.path.join(main_dir, data_info["annotations"]))
    arrows = df["Type"] == "Arrow"
    x_coord = df.loc[arrows]["X"]
    y_coord = df.loc[arrows]["Y"]

    # They are not many annotations, let's just do a loop for the moment:
    # TODO: possibly, I could use isin and an index matrix?
    for x, y in zip(x_coord, y_coord):
        # Just insert some high number:
        # print(y,x)
        binary_mask[0, y, x] = 4000

    print(binary_mask.max())
    print((binary_mask == 4000).sum())
    out_path = os.path.join(main_dir, data_info["input_image"]).replace(".tif", "_annotation_mask.ome.tif")
    tiff.imwrite(out_path, binary_mask, metadata={'axes': "CYX"})


