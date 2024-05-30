import numpy as np

def convert_predictions_to_panoptic_segm(predictions):
    instances = predictions['instances']

    shape = instances.image_size
    masks = instances.pred_masks.to("cpu").numpy()
    box_scores = instances.scores.to("cpu").numpy()
    # confidence_scores = instances.mask_scores.to("cpu").numpy()

    # Sort masks according to their score:
    panoptic_segm = np.zeros(shape)
    argsprt = np.argsort(box_scores)
    masks = masks[argsprt]

    # TODO: find way to avoid python loop
    for i in range(box_scores.shape[0]):
        panoptic_segm[masks[i]] = i+1

    # TODO: add IoU threshold
    # TODO: run connected components..?

    return panoptic_segm

