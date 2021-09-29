import os

import shutil
from segmfriends.utils.various import check_dir_and_create


def infer_LIVECell_directory(in_dir, out_dir):
    # TODO: generalize to other models and configs
    config_path = "/scratch/bailoni/pyCh_repos/LIVECell/model/anchor_free/livecell_config.yaml"

    shutil.rmtree(out_dir)
    check_dir_and_create(out_dir)

    image_names = ""
    for root, dirs, files in os.walk(in_dir):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
                if not filename.startswith("."):
                    image_names += "{} ".format(os.path.join(root, filename))
    command = "ipython segmUtils/segmentation/LIVECell/infer-centermask.py --  --input {} --output {} --config-file {}".format(
        image_names, out_dir, config_path
    )
    os.system(command)
    # stream = os.popen(command)
    # print(stream.read())


if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"
    input_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images_small/LIVECell")
    out_dir = os.path.join(scratch_dir, "projects/spacem_segm/segm/LIVECell")
    infer_LIVECell_directory(input_dir, out_dir)
