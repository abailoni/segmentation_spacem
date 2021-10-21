import numpy as np
from cellpose.metrics import aggregated_jaccard_index, average_precision

from inferno.io.volumetric.volumetric_utils import slidingwindowslices
import os
import json, cv2, random
import imageio
import shutil
from copy import deepcopy
import pandas as pd
import vigra

from segmfriends.utils.various import check_dir_and_create


def compute_scores(pred_dir, GT_dir,
                           pred_filter="_cp_masks", GT_filter="_masks",
                            AP_thresholds=(0.5,0.75,0.9)):
    scores = {
        'aji': [],
        'ap': [],
        'tp': [],
        'fp': [],
        "fn": []
    }
    # i = 0
    assert os.path.exists(pred_dir)
    assert os.path.exists(GT_dir)
    for root, dirs, files in os.walk(pred_dir):
        for filename in files:
            file_basename, file_extension = os.path.splitext(filename)
            assert file_extension == ".png", "Atm, pred and GT masks are expected to be in png format"
            if file_basename.endswith(pred_filter):
                # Look for corresponding GT masks:
                GT_masks_filename = filename.replace(pred_filter, GT_filter)
                GT_masks_path = os.path.join(GT_dir, GT_masks_filename)
                assert os.path.exists(GT_masks_path), "GT file not found!"

                # Load masks:
                pred_masks = imageio.imread(os.path.join(root, filename))
                GT_masks = imageio.imread(GT_masks_path)

                # availableFeatures = vigra.analysis.supportedRegionFeatures(pred_masks.astype('float32'), pred_masks.astype('uint32'))
                # skeletonFeatures = vigra.analysis.extractSkeletonFeatures(pred_masks.astype('uint32'), pruning_threshold=0.2)
                # # TODO: only compute radii
                # feature_extractor = vigra.analysis.extractRegionFeatures(pred_masks.astype('float32'), pred_masks.astype('uint32'), features= ['RegionRadii', 'RegionAxes', 'RegionCenter',
                #                                                                                            # 'Weighted<RegionCenter>', 'Weighted<RegionRadii>', 'Weighted<RegionAxes>' These return the same values...
                #                                                                                                                                ],ignoreLabel=0)
                #
                # # Ignore background label in the mean/max/min statistics:
                #
                # print("Diameter mean: ", skeletonFeatures["Diameter"][1:].mean())
                # # TODO: this looks like a good estimate!
                # print("EDiameter mean: ", skeletonFeatures["Euclidean Diameter"][1:].mean())
                # # print(skeletonFeatures["Average Length"][1:].mean())
                # # print(skeletonFeatures["Total Length"][1:].mean())
                #
                # mean_large_radius, mean_small_radius = feature_extractor['RegionRadii'][1:].mean(axis=0)
                # print("Large-Radii mean: ", feature_extractor['RegionRadii'][1:].mean(axis=0))
                # print("Large-Radii max: ", feature_extractor['RegionRadii'][1:].max(axis=0))
                # # print(feature_extractor['RegionRadii'][1:].min(axis=0))

                # Compute scores:
                aji_current = aggregated_jaccard_index([GT_masks], [pred_masks])
                ap_c, tp_c, fp_c, fn_c = average_precision(GT_masks, pred_masks, threshold=list(AP_thresholds))
                if np.any(np.isnan(aji_current)):
                    print("")
                scores['aji'].append(aji_current)
                scores['ap'].append(ap_c)
                scores['tp'].append(tp_c)
                scores['fp'].append(fp_c)
                scores['fn'].append(fn_c)
                # i += 1
                # if i > 5:
                #     break

        # Only look in top directory
        break
    # Average scores:
    assert len(scores['aji']) > 0, "No images found in given folders"
    for score_type in scores:
        scores[score_type] = np.array(scores[score_type]).mean(axis=0)
    return scores




if __name__ == "__main__":
    scratch_dir = "/scratch/bailoni"

    # ------------------------------
    # Few sample and cropped images:
    # ------------------------------
    # input_dir = os.path.join(scratch_dir, "datasets/LIVECell/panoptic/livecell_coco_test")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1_LIVECelltest")
    # input_dir = os.path.join(scratch_dir, "datasets/cellpose/test")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1_cellpose_test")
    # input_dir = os.path.join(scratch_dir, "projects/spacem_segm/alex_labeled")
    # out_dir = os.path.join(scratch_dir, "projects/train_cellpose/predictions/test/model1_alex")

    models_to_test = ["cyto2_diamEst",
                      "cyto_diamEst",
                      "trained_on_LIVECell_noDiamEst",
                      "trained_on_cellpose_noDiamEst",
                      "finetuned_LIVECell_lr_02_noDiamEst",
                      "finetuned_LIVECell_lr_00002_noDiamEst",
                      "trained_on_LIVECell_diamEst",
                      "trained_on_cellpose_diamEst",
                      "finetuned_LIVECell_lr_02_diamEst",
                      # "finetuned_LIVECell_lr_00002_diamEst",
                      ]

    dirs_to_process = [
        [
            os.path.join(scratch_dir, "datasets/alex/labels"),
            os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/alex")
        ],
        [
            os.path.join(scratch_dir, "datasets/cellpose/test"),
            os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/cellpose_test")
        ],
        [
            os.path.join(scratch_dir, "datasets/LIVECell/panoptic/livecell_coco_test"),
            os.path.join(scratch_dir, "projects/train_cellpose/predictions/$MODEL_NAME/LIVECell_test")
        ],
    ]

    collected_scores_names = None
    collected_scores = []
    for model_name in models_to_test:
        for GT_dir, pred_dir in dirs_to_process:
            pred_dir = pred_dir.replace("$MODEL_NAME", model_name)

            AP_thresholds = (0.5, 0.75, 0.9)
            scores = compute_scores(pred_dir, GT_dir, AP_thresholds=AP_thresholds)

            # Prepare scores to be written to csv file:
            scores_names = []
            new_collected_scores = []
            for sc_name in scores:
                score = scores[sc_name]
                assert len(score.shape) == 1
                for AP_thr_indx, scr in enumerate(score):
                    new_collected_scores.append(scr)
                    if collected_scores_names is None:
                        if score.shape[0] == 1:
                            scores_names.append(sc_name)
                        else:
                            assert score.shape[0] == len(AP_thresholds)
                            scores_names.append("{}_{}".format(sc_name, AP_thresholds[AP_thr_indx]))
            if collected_scores_names is None:
                collected_scores_names = deepcopy(scores_names)
            basedir = os.path.basename(os.path.normpath(pred_dir))

            if "_noDiamEst" in model_name:
                estimate_diam = 0
            elif "_diamEst" in model_name:
                estimate_diam = 1
            else:
                estimate_diam = None

            collected_scores.append([basedir, model_name, estimate_diam] + new_collected_scores)
            print("Done {}, {}".format(model_name, pred_dir))



    df = pd.DataFrame(collected_scores, columns=['Data type', 'Model name', 'Estimated cell size'] + collected_scores_names)
    df.sort_values(by=['Data type', 'aji'], inplace=True, ascending=False)
    df.to_csv("/scratch/bailoni/projects/train_cellpose/scores_with_estimated_diameters_all.csv")




