import os
import numpy as np
from shutil import copyfile


dataset_dir = "/Volumes/ExtremeSSD/datasets/ATeam/cellpose_data_original"
output_dataset = "/Volumes/ExtremeSSD/datasets/ATeam/cellpose_data_restructured"




def check_dir_and_create(directory):
    '''
    if the directory does not exist, create it
    '''
    folder_exists = os.path.exists(directory)
    if not folder_exists:
        os.makedirs(directory)
    return folder_exists

check_dir_and_create(output_dataset)

for dataset in ["train", "test"]:
    # Create directories:
    out_dir = os.path.join(output_dataset, dataset)
    out_dir_img = os.path.join(output_dataset, dataset, "images")
    out_dir_labels = os.path.join(output_dataset, dataset, "labels")
    check_dir_and_create(out_dir)
    check_dir_and_create(out_dir_img)
    check_dir_and_create(out_dir_labels)

    for root, dirs, files in os.walk(os.path.join(dataset_dir, dataset)):
        for filename in files:
            if filename.endswith(".png") and not filename.startswith(".") and "_img" in filename:
                # Get corresponding mask file:
                labels_file = filename.replace("_img.png", "_masks.png")

                # Get output paths:
                base_out_name = filename.replace("_img", "")
                copyfile(os.path.join(root, filename), os.path.join(out_dir_img, base_out_name))
                copyfile(os.path.join(root, labels_file), os.path.join(out_dir_labels, base_out_name))







