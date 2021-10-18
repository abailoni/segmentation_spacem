import os

if __name__ == "__main__":

    # --------------------------
    # CellPose data + LIVECell data:
    # --------------------------
    train_folder = "/scratch/bailoni/projects/train_cellpose/data/train"
    test_folder = "/scratch/bailoni/projects/train_cellpose/data/test"
    # train_folder = "/scratch/bailoni/projects/train_cellpose/data/few_images_train"
    # test_folder = "/scratch/bailoni/projects/train_cellpose/data/few_images_test"
    pretrained_model = "cyto2"
    # pretrained_model = "/scratch/bailoni/projects/train_cellpose/data/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_10_11_16_44_29.601896"
    save_every = 20
    # initial_learning_rate = 0.2  # 0.0002
    initial_learning_rate = 0.0002  # 0.0002


    # # --------------------------
    # # Only cellPose data:
    # # --------------------------
    # train_folder = "/scratch/bailoni/datasets/cellpose/train"
    # test_folder = "/scratch/bailoni/datasets/cellpose/test"
    # pretrained_model = None
    # save_every = 50
    # # save_dir = "/scratch/bailoni/projects/train_cellpose/models/only_cellpose_data"
    # initial_learning_rate = 0.2  # 0.0002

    first_ch = 2
    second_ch = 1
    nb_epochs = 500 # 500

    batch_size = 8

    mask_filter = "_masks"

    # # Check if label files are consistent:
    # for root, dirs, files in os.walk(train_folder):
    #     for file in files:
    #         if

    command = "ipython -m cellpose -- --train --use_gpu --fast_mode --dir {} --test_dir {} --pretrained_model {} " \
              "--chan {} --chan2 {} --n_epochs {} --learning_rate {} --batch_size {} " \
              "--no_npy --mask_filter {} --save_every {} --verbose --train_size".format(
        train_folder,
        test_folder,
        pretrained_model,
        first_ch,
        second_ch,
        nb_epochs,
        initial_learning_rate,
        batch_size,
        mask_filter,
        save_every,
    )
    os.system(command)
    print("Learning rate: ", initial_learning_rate)

