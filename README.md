# Segmentation SpaceM
A very "early-development-stage" set of utilities to segment images using Cellpose and other tools.

## Installation
1. First, we create the conda environment. You will need PyTorch with cuda support in order to use Cellpose with GPU support. Instructions to install PyTorch could change over time, so you are welcome to double-check [here](https://pytorch.org/get-started/locally/). Since PyTorch with CUDA takes a lot of space, you may need to install the environment in a different location than the default one (for example on a scratch folder, but it will likely be deleted after a while). First, let's see the command to install it in the default location: 

   ```bash
   conda create -n segmSpaceM python=3.8 pytorch torchvision torchaudio segmfriends pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -c abailoni
   conda activate segmSpaceM 
   ```

   If you want to install it in a different location, you can use the following command:

   ```bash
   conda create --prefix /path/to/folder/of/your/new/conda/env python=3.8 pytorch torchvision torchaudio segmfriends pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -c abailoni
   conda activate /path/to/folder/of/your/new/conda/env
   ```

2. In order to install the package, you will need access to some SpaceM packages. Sign-in to GitLab and create a
   [personal access token](gitlab:/-/profile/personal_access_tokens) with scope `read_api`,
   or ask a SpaceM developer to provide access for you.

3. Create a file `pip.config` (or `pip.ini`) at the location specified in the
   [pip documentation](https://pip.pypa.io/en/stable/topics/configuration/#location) with the
   following content, replacing `TOKENNAME` and `TOKEN` with the respective values:

   ```ini
   [global]
   # Be aware that pip will also search in --index-url (https://pypi.org/simple/).
   # If package names clash, it will not prioritize our private repository, but the higher version number.
   extra-index-url = https://TOKENNAME:TOKEN@git.embl.de/api/v4/groups/1245/-/packages/pypi/simple
   ```
   
4. Finally, we clone the repository and install the package:

   ```bash
   git clone https://github.com/abailoni/segmentation_spacem.git
   cd segmentation_spacem
   pip install -e .
   ```

## Usage
### Predicting Cellpose segmentations in batch mode
In the `example_project` directory of the repository you find an example of script and config file you may need to process some SpaceM datasets and predict cellpose segmentations for each of them. Create a copy of such folder and adapt the content to your needs:

1. In the `infer_cellpose.py` script you will need to specify the path where you want to save the results.
2. In the config file you will find comments guiding you on how to specify the paths to the images you want to process, which cellpose models you wish to run, etc.
3. Once you exported the results, you will find the segmentations in the `exported_results` subdirectory of the path you specified in the `infer_cellpose.py` script.

### Visualizing all microscopy images and segmentations in neuroglancer
1. First, install neuroglancer by running `pip install neuroglancer`, after activating the conda environment.
2. Use the `visualize-data-in-neuroglancer.ipynb` notebook in the `example_project` directory to start the neuroglancer server and load your results (instructions are present in the notebook).
