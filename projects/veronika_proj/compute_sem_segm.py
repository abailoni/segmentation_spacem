import os.path

import numpy as np
import vigra
import segmfriends.io.zarr as zarr_utils
import nifty.graph.rag
import nifty.tools as ntools

if __name__ == "__main__":
    proj_dir = "/scratch/bailoni/datasets/veronika/macrophages_Bosurgi6"

    prediction_file = "cellpose_GFP_DAPI_full_images/predictions/predictions.zarr"

    # TODO: do it for every slice if too memory consuming...?

    # Import raw data:
    mCherry_ch = zarr_utils.load_array_from_zarr_group(
        os.path.join(proj_dir, "data.zarr"),
        "mCherry"
    )

    # # Import segmentations:
    # mCherry_segm = zarr_utils.load_array_from_zarr_group(
    #     os.path.join(proj_dir, "cellpose_mCherry/predictions/predictions.zarr"),
    #     "cyto2_diamEst"
    # )
    GFP_segm = zarr_utils.load_array_from_zarr_group(
        os.path.join(proj_dir, prediction_file),
        "cyto2_diamEst"
    )

    # Compute mCherry segmentation:
    mCherry_mask = mCherry_ch >= 13
    # # Get segments?
    # for z in range(mCherry_mask.shape[0]):
    #     pass
    #     vigra.analysis.labelImage()

    rag = nifty.graph.rag.gridRag(GFP_segm.astype('uint32'))
    _, node_feat = nifty.graph.rag.accumulateMeanAndLength(rag, mCherry_ch.astype('float32'))
    # node_feat = nifty.graph.rag.accumulateNodeStandartFeatures(rag, mCherry_ch.astype('float32'), minVal=0., maxVal=255.)
    mean_mCherry_values = node_feat[:, [0]]

    mean_mCherry_values[np.isnan(mean_mCherry_values)] = 0

    # Set segments with mCheery-mean > 2.0 as active:
    sem_mask = np.ones_like(mean_mCherry_values)
    sem_mask[mean_mCherry_values >= 2.0] = 2.

    # mapped_feat = ntools.mapFeaturesToLabelArray(GFP_segm, mean_mCherry_values, ignore_label=0, fill_value=-1)[...,0]
    mapped_sem_segm = ntools.mapFeaturesToLabelArray(GFP_segm, sem_mask, ignore_label=0, fill_value=0)[..., 0].astype(
        'uint16')

    mapped_sem_segm[np.logical_and(mCherry_mask, mapped_sem_segm == 0)] = 3

    zarr_utils.add_dataset_to_zarr_group(
        os.path.join(proj_dir, prediction_file),
        mapped_sem_segm,
        "sem_segmentation",
        add_array_dimensions=True
    )
