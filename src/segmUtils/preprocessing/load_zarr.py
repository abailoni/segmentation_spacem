# import traincellpose.io.ome_zarr_utils as ome_zarr_utils
# ome_zarr_utils.load_ome_zarr_channels(ome_zarr_path="/scratch/bailoni/projects/cellpose_training_projects/luisa/2022_oct_seahorse/data/slide1/microscopy.zarr/A1",
#                                       )
from ngff_writer.writer import open_ngff_image, open_ngff_collections

import os
# dir_filter = "microscopy.zarr"
optical_image_channel = "Trans"
metadata_path = "/scratch/bailoni/projects/cellpose_training_projects/luisa/2022_oct_seahorse/well_metadata.csv"
out_format = "/scratch/bailoni/projects/cellpose_training_projects/luisa/2022_oct_seahorse/data/tif_images/slide{slide}/{treatment}/{well_id}_{treatment}_trans_ch_0.tif"
input_dir_format = "/scratch/bailoni/projects/cellpose_training_projects/luisa/2022_oct_seahorse/data/slide{slide}/microscopy.zarr/{well_id}/pre_maldi"



from traincellpose.io.images import write_image_to_file

from PIL import Image, ImageTransform

import pandas as pd
metadata = pd.read_csv(metadata_path)

for index, row in metadata.iterrows():
    full_path = input_dir_format.format(**row.to_dict())
    print(f"Loading and writing {full_path}...")
    img_pre = open_ngff_image(full_path)

    channel_index_pre = img_pre.channel_names.index(optical_image_channel)

    # # cell segmentation
    # img_cells = img_pre.labels["cells"]
    # tform_cells = img_pre.labels["cells"].transformation[-3:, -3:]

    # # make tform translation relative to tform_pre
    # tform_post[..., -1] -= tform_pre[..., -1]
    # tform_post[..., -1] *= -1  # honestly don't know why this has to be done
    #
    # tform_cells[:, -1] -= tform_pre[:, -1]
    #
    # tform_pre[..., -1] = 0

    # convert to PIL, apply affine transformations to match fiducial registration
    img_array = img_pre.array()[0, channel_index_pre, 0]

    full_out_path = out_format.format(**row.to_dict())
    out_dir, _ = os.path.split(full_out_path)
    os.makedirs(out_dir, exist_ok=True)
    write_image_to_file(full_out_path, img_array)


# img_pre = open_ngff_collections("/scratch/bailoni/projects/cellpose_training_projects/luisa/2022_oct_seahorse/data/slide1/microscopy.zarr")
# print(img_pre)


# print(ome_zarr_utils.get_channel_list_in_ome_zarr(ome_zarr_path=""
#                                       ))
