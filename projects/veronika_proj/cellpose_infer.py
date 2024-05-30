import os.path
import sys

import cv2
import numpy as np
import pandas
import vigra
import segmfriends.io.zarr as zarr_utils
import nifty.graph.rag
import nifty.tools as ntools

from segmUtils.io.export_images_from_zarr import export_images_from_zarr
from segmUtils.preprocessing.preprocessing import convert_images_to_zarr_dataset, read_uint8_img, \
    read_segmentation_from_file
from segmfriends.io.images import write_segm_to_file
from speedrun import process_speedrun_sys_argv
from segmfriends.utils import check_dir_and_create

from speedrun import BaseExperiment
from segmfriends.utils.paths import get_vars_from_argv_and_pop

from segmUtils.segmentation.cellpose.base_experiment import CellposeBaseExperiment

from segmUtils.preprocessing import preprocessing as spacem_preproc
from segmUtils.segmentation.cellpose import infer as cellpose_infer
from segmUtils.postprocessing.convert_to_zarr import convert_segmentations_to_zarr, \
    convert_multiple_cellpose_output_to_zarr



class MacrophagesExperiment(CellposeBaseExperiment):
    def compute_semantic_segm(self):
        # TODO: do it for every slice if too memory consuming...?
        zarr_path_predictions = self.zarr_path_predictions
        input_zarr_group_path = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        sem_segm_kwargs = self.get("compute_semantic_segm/compute_semantic_segm_kwargs",
                                   ensure_exists=True)

        # Import raw data:
        mCherry_ch = zarr_utils.load_array_from_zarr_group(
            input_zarr_group_path,
            sem_segm_kwargs.get("red_channel_name", "mCherry")
        )

        print("Computing semantic segmentation...")
        for segm_data in sem_segm_kwargs.get("instance_segm_to_process", []):
            GFP_segm = zarr_utils.load_array_from_zarr_group(
                zarr_path_predictions,
                segm_data["instance_segm_path"]
            )
            GFP_segm, _, _ = vigra.analysis.relabelConsecutive(GFP_segm.astype("uint32"))
            # Compute mCherry segmentation:
            mCherry_mask = mCherry_ch >= sem_segm_kwargs.get("red_ch_thresh", 13)

            # # Get segmentation:
            # mCherry_segm = np.zeros_like(mCherry_mask, dtype="uint32")
            # max_label = 0
            # for z in range(mCherry_mask.shape[0]):
            #     mCherry_segm[z] = vigra.analysis.labelImage(mCherry_mask[z]) + max_label
            #     max_label += mCherry_segm[z].max() + 1

            # TODO: check for nan values in the segmentation (and corner)
            rag = nifty.graph.rag.gridRag(GFP_segm.astype('uint32'))
            _, node_feat = nifty.graph.rag.accumulateMeanAndLength(rag, mCherry_mask.astype('float32'))
            # node_feat = nifty.graph.rag.accumulateNodeStandartFeatures(rag, mCherry_ch.astype('float32'), minVal=0., maxVal=255.)
            mean_mCherry_values = node_feat[:, [0]]
            assert np.isnan(mean_mCherry_values).sum() == 0, "Something went wrong"
            size_macrophages = node_feat[:, [1]]
            size_eaten_mCherry = mean_mCherry_values*size_macrophages
            # mean_mCherry_values[np.isnan(mean_mCherry_values)] = 0

            # Set segments with mCheery-mean > 2.0 as active:
            sem_mask = np.ones_like(mean_mCherry_values)
            sem_mask[size_eaten_mCherry >= sem_segm_kwargs.get("size_eaten_mCherry_thresh", 10)] = 2.

            # mapped_feat = ntools.mapFeaturesToLabelArray(GFP_segm, mean_mCherry_values, ignore_label=0, fill_value=-1)[...,0]
            mapped_sem_segm = ntools.mapFeaturesToLabelArray(GFP_segm, sem_mask, ignore_label=0, fill_value=0)[
                ..., 0].astype(
                'uint16')

            # Semantically label small cherry segments:
            mapped_sem_segm[np.logical_and(mCherry_mask, mapped_sem_segm == 0)] = 3

            # Delete small segments and get final segmentation:
            final_segm = np.where(GFP_segm == 0, mapped_sem_segm, GFP_segm + 5)
            max_label = 0
            for z in range(mapped_sem_segm.shape[0]):
                background_mask = final_segm[z] == 0
                final_segm[z] = vigra.analysis.labelImageWithBackground(final_segm[z].astype('uint32')) + max_label
                final_segm[z][background_mask] = 0 # TODO: not necessary anymore
                max_label += final_segm[z].max() + 1

            def get_size_map(label_image):
                node_sizes = np.bincount(label_image.flatten())
                return ntools.mapFeaturesToLabelArray(label_image, node_sizes[:, None], nb_threads=6).squeeze()

            size_map = get_size_map(final_segm)

            print("Size threshold: {}".format(sem_segm_kwargs.get("size_threshold", 25)))
            mask_small_segments = size_map < sem_segm_kwargs.get("size_threshold", 25)
            mapped_sem_segm[mask_small_segments] = 0
            final_segm[mask_small_segments] = 0

            # Save segmentations to zarr file:
            zarr_utils.add_dataset_to_zarr_group(
                zarr_path_predictions,
                mapped_sem_segm,
                "sem_segmentation",
                add_array_dimensions=True
            )

            zarr_utils.add_dataset_to_zarr_group(
                zarr_path_predictions,
                final_segm,
                "final_segmentation",
                add_array_dimensions=True
            )

        print("Done processing semantic segmentations")

    def export_results(self):
        """
        For the moment this method is thought for inference. Generalize...?
        """
        zarr_path_predictions = self.zarr_path_predictions
        input_zarr_group_path = self.get("preprocessing/data_zarr_group", ensure_exists=True)
        export_images_from_zarr_kwargs = self.get("export_results/export_images_from_zarr_kwargs", ensure_exists=True)
        csv_config_path = input_zarr_group_path.replace(".zarr", ".csv")

        export_dir = os.path.join(self.experiment_directory, "exported_results")
        self.set("export_results/export_path", export_dir)

        # Insert zarr path of the prediction file in the export parameters:
        assert "datasets_to_export" in export_images_from_zarr_kwargs
        datasets_to_export = export_images_from_zarr_kwargs.pop("datasets_to_export")
        for idx in range(len(datasets_to_export)):
            datasets_to_export[idx]["z_path"] = zarr_path_predictions

        print("Exporting result images to original folder structure...")


        # Export images in the original structure:
        export_images_from_zarr(export_dir,
                                csv_config_path,
                                datasets_to_export=datasets_to_export,
                                delete_previous=True,
                                **export_images_from_zarr_kwargs)

    def convert_sem_segm_to_csv(self):
        export_dir = os.path.join(self.experiment_directory, "exported_results")

        print("Converting semantic segmentation to csv format...")
        df = convert_images_to_zarr_dataset(input_dir_path=export_dir,
                                            # ensure_all_channel_existance=True,
                                            rename_unique=False,
                                            extension=".tif",
                                            save_to_zarr=False,
                                            verbose=True,
                                            final_segm="_cell_segm",
                                            sem_segm="_cell_type"
                                            )

        for idx, image_data in df.iterrows():
            dir = image_data["Input dir"]
            final_segm = read_segmentation_from_file(os.path.join(dir, image_data["final_segm"]))
            sem_segm = read_segmentation_from_file(os.path.join(dir, image_data["sem_segm"]))

            print("Max sem segmentation for {}: {}".format(image_data["final_segm"], sem_segm.max()))

            # Save semantic segmentation to csv:
            final_segm, _, _ = vigra.analysis.relabelConsecutive(final_segm.astype("uint32"))

            rag = nifty.graph.rag.gridRag(final_segm.astype('uint32'))
            _, node_feat = nifty.graph.rag.accumulateMeanAndLength(rag, sem_segm.astype('float32'))
            node_semantic_segmentation = node_feat[:, 0].astype('uint32')

            # Write semantic data to csv:
            df = pandas.DataFrame({'cell_instance_ID': np.arange(1, node_semantic_segmentation.shape[0]),
                                   'semantic_class': node_semantic_segmentation[1:]})
            df.to_csv(os.path.join(dir,image_data["sem_segm"].replace(".tif", ".csv")),index=False, sep=";")

            # Rewrite the new relabelled segmentation:
            write_segm_to_file(os.path.join(dir, image_data["final_segm"]), final_segm)

    def test_enhance(self):
        in_dir = os.path.join(self.experiment_directory, "cellpose_inputs/GFP_DAPI")

        df = convert_images_to_zarr_dataset(input_dir_path=in_dir,
                                            rename_unique=False,
                                            extension=".png",
                                            save_to_zarr=False,
                                            verbose=True,
                                            cellpose_input="_0",
                                            )

        from PIL import Image, ImageEnhance

        for idx, image_data in df.iterrows():
            dir = image_data["Input dir"]
            cellpose_input = read_uint8_img(os.path.join(dir, image_data["cellpose_input"]))
            pil_img = Image.fromarray(cellpose_input)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img_out = enhancer.enhance(3)

            out_dir = os.path.join(dir, "../bright")
            check_dir_and_create(out_dir)
            pil_img_out.save(os.path.join(out_dir, image_data["cellpose_input"]))
            # cv2.imwrite(os.path.join(out_dir, image_data["cellpose_input"]), pil_img_out)
            break



if __name__ == '__main__':
    source_path = os.path.dirname(os.path.realpath(__file__))
    sys.argv = process_speedrun_sys_argv(sys.argv, source_path, default_config_dir_path="./configs",
                                         default_exp_dir_path="/scratch/bailoni/projects/cellpose_projects")

    cls = MacrophagesExperiment
    cls().run()
