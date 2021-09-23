import os
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/scratch/bailoni/.cellpose/models"

from cellpose import models
import skimage.io


# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto')


"""
python -m cellpose --dir /scratch/bailoni/projects/CellPose/test_inference --pretrained_model cyto --chan 0 --save_png --use_gpu

"""
