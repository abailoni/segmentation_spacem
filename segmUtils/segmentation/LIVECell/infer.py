import os

import shutil

import argparse

from segmfriends.utils.various import check_dir_and_create


def infer_LIVECell_directory(in_dir, out_dir, model="anchor-free"):
    # TODO: generalize to other models and configs
    if model == "anchor-free":
        config_path = "/scratch/bailoni/pyCh_repos/LIVECell/model/anchor_free/livecell_config.yaml"
    elif model == "anchor-based":
        config_path = "/scratch/bailoni/pyCh_repos/LIVECell/model/anchor_based/livecell_config.yaml"
    else:
        raise ValueError(model)

    # Create or erase out-dir:
    out_dir = os.path.join(out_dir, model)
    check_dir_and_create(out_dir)
    shutil.rmtree(out_dir)
    check_dir_and_create(out_dir)

    image_names = ""
    for root, dirs, files in os.walk(in_dir):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
                if not filename.startswith("."):
                    image_names += "{} ".format(os.path.join(root, filename))
                    infer_script = "infer-centermask.py" if model == "anchor-free" else "infer-anchor-based.py"
                    command = "ipython segmUtils/segmentation/LIVECell/infer-centermask.py --  --input {} --output {} --config-file {}".format(
                        os.path.join(root, filename), out_dir, config_path
                    )
                    os.system(command)
    # stream = os.popen(command)
    # print(stream.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIVECell inference")
    # parser.add_argument(
    #     "--config-file",
    #     default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    # )
    parser.add_argument("--model", type=str, help="Type of LIVECell model (anchor-based or anchor-free)", default="anchor-free")
    # parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #          "If not given, will show output in an OpenCV window.",
    # )
    args = parser.parse_args()
    model = args.model
    assert model in ["anchor-based", "anchor-free"]

    scratch_dir = "/scratch/bailoni"
    input_dir = os.path.join(scratch_dir, "projects/spacem_segm/input_images/LIVECell")
    out_dir = os.path.join(scratch_dir, "projects/spacem_segm/segm/LIVECell")
    # input_dir = os.path.join(scratch_dir, "projects/spacem_segm/original_LIVECell_images")
    # out_dir = os.path.join(scratch_dir, "projects/spacem_segm/segm/original_LIVECell_images")
    infer_LIVECell_directory(input_dir, out_dir, model)

