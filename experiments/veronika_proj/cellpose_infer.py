import os.path
import sys

import numpy as np
import vigra
import segmfriends.io.zarr as zarr_utils
import nifty.graph.rag
import nifty.tools as ntools

from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmfriends.speedrun_exps.utils import process_speedrun_sys_argv

from speedrun import BaseExperiment
from segmfriends.utils.paths import get_vars_from_argv_and_pop

from segmUtils.segmentation.cellpose.base_experiment import CellposeBaseExperiment

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, convert_multiple_cellpose_output_to_zarr


class MacrophagesExperiment(CellposeBaseExperiment):
    def compute_semantic_segm(self):
        # TODO: do it for every slice if too memory consuming...?

        zarr_path_predictions = self.get("cellpose_inference/zarr_path_predictions", ensure_exists=True)
        input_zarr_group_path = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        sem_segm_kwargs = self.get("compute_semantic_segm/compute_semantic_segm_kwargs",
                                                ensure_exists=True)


        # Import raw data:
        mCherry_ch = zarr_utils.load_array_from_zarr_group(
            input_zarr_group_path,
            sem_segm_kwargs.get("red_channel_name", "mCherry")
        )

        for segm_data in sem_segm_kwargs.get("instance_segm_to_process", []):
            GFP_segm = zarr_utils.load_array_from_zarr_group(
                zarr_path_predictions,
                segm_data["instance_segm_path"]
            )

            # Compute mCherry segmentation:
            mCherry_mask = mCherry_ch >= sem_segm_kwargs.get("red_ch_thresh", 13)

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
            sem_mask[mean_mCherry_values >= sem_segm_kwargs.get("red_ch_cell_mean_thresh", 2.)] = 2.

            # mapped_feat = ntools.mapFeaturesToLabelArray(GFP_segm, mean_mCherry_values, ignore_label=0, fill_value=-1)[...,0]
            mapped_sem_segm = ntools.mapFeaturesToLabelArray(GFP_segm, sem_mask, ignore_label=0, fill_value=0)[
                ..., 0].astype(
                'uint16')

            mapped_sem_segm[np.logical_and(mCherry_mask, mapped_sem_segm == 0)] = 3

            zarr_utils.add_dataset_to_zarr_group(
                zarr_path_predictions,
                mapped_sem_segm,
                segm_data["out_zarr_inner_path"],
                add_array_dimensions=True
            )


if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path, default_config_rel_path="./configs")

    cls = MacrophagesExperiment
    cls().run()
