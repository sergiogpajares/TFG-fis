# VGG cloud NET

This is the codebase on which my Honour's Project (in spanish Trabajo de Fin de Grado) has been built on.

The project creates and trains a Convolutional Neural Networks for multi-class semantic segmentation of whole-sky ground images for scientific proposes.

## Installation guide

Installation is given for a UNIX-like system. On a windows machine, the setup would be similar but some of the given commands will be different. For development, Ubuntu 22.04.4 has been used.

> [!IMPORTANT]
> To benefit the most, you should have an NVIDIA graphic card properly setup on your computer.
> If you're not sure weather you have one, run `nvidia-smi`.
> Also, you'll need to have `CuDNN 8.9.9` or higher. At the time of writing `CuDNN 4.9.0` is not supported by TensorFlow. `TensorRT`is recommended but not required. I have not used it.

The code has been tested and developed with `Python 3.10.12`. If you're using a different python version, the code will most likely work. This python version can be installed with

```bash
sudo apt-get install python3.10 --assume-yes
``` 

The project can be easily install on a virtual environment. For that, `venv` python package is required. It can be installed with
```bash
sudo apt-get install python3.10-venv --assume-yes
```

To create a Virtual Environment called `venv` run
```bash
# At the project folder
python3.10 -m venv venv
```

The virtual environment can be activated with

```bash
source venv/bin/activate
```
and deactivated with `deactivate`. Most IDEs, like VS code, will automatically open the venv on project opening.

Dependencies can be installed from the `requirements.txt`. But, `opencv>=4.9.0` is not provided in this file.
> [!NOTE]
> **OpenCV 4.9.0** If you want to benefit from GPU accelerated computations on OpenCV, you must compile it from scratch with CUDA and CuDNN support. Please refer to [OpenCV Installation Guide](assets/libraries/opencv_install_guide.md) to find guidance on it. If using your own 

```bash
pip install -r requirements.txt
```

### Using LaTeX with MatPlotLib

To be able to use LaTeX along with `Matplotlib` some additional system packages are requiered. I use `texlive` LaTeX distributions. Some windows users may prefer using `miktex` distribution.
```bash
sudo apt-get install texlive-latex-extra dvipng cm-super --assume-yes
```
Also, if using language support
```bash
sudo apt-get install texlive-lang-spanish
```

If you don't want LaTeX to be used, please comment the following lines at the top of the code files
```python
matplotlib.rcParams.update({
    'text.usetex' : True,
    'text.latex.preamble' : r'\usepackage[spanish,es-tabla]{babel}\decimalpoint\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}'
}) 
```

## Project structure
The project have the following folder structure

```
TFG-fis
 |
 ├── assets
 |    └── libraries 
 |          └── opencv_install_guide.md
 ├── dataset
 |    ├── images
 |    ├── masks
 |    ├── augmented_images_train
 |    ├── augmented_images_val
 |    ├── augmented_masks_train
 |    ├── augmented_masks_val
 |    ├── train.txt 
 |    ├── val.txt
 |    └── test.txt
 ├── logs
 ├── models
 |    ├── VGG16unet-224x224-50epochs.model.keras
 |    └── Checkpoints
 ├── utils
 |     ├── __init__.py
 |     └── loading.py
 ├── trainVGG16unet.ipynb
 ├── metrics.ipynb
 ├── augmentation.py
 ├── mask_painter.py 
 ├── README.md
 ├── LICENSE
 └── .gitignore
```
> [!ALERT]
> The **dataset** is not publicly available at the moment. It is property of the [GOA](https://goa.uva.es/) research group. More about them at their webpage. You can also see the stations and some realtime images in their [webpage](https://goa.uva.es/proyecto-presente/). `train.txt`, `val.txt` and `test.txt` contain a list of filename to be used for training, validation and testing respectively.

> [!ALERT]
> A pre-trained **model** can be found in [kaggle](https://www.kaggle.com/models/sergiogarciapajares/vggcloudunet). It can't be found within the github respository (due to size limitations).

## Usage guide

### Data augmentation
`utils/augmentation.py` is a CLI util to augment images based on the `albumentations` library

```
usage: augmentation.py [-h] [-n NIMAGES] [--limit LIMIT] [--img-extension {jpg,png}] [--mask-extension {jpg,png}]
                       [--n-threads N_THREADS] [--remove]
                       {train,val}

Generate augmented images for train or validation according to mocked into the src code

positional arguments:
  {train,val}           Wheater to generate train or val augmented images

options:
  -h, --help            show this help message and exit
  -n NIMAGES, --nimages NIMAGES
                        Number of images to generate from each original one
  --limit LIMIT         Maximun number of original images to use. This option is intended for test and debug
  --img-extension {jpg,png}
                        Extension to save augmented images (not masks)
  --mask-extension {jpg,png}
                        Extension to save augmented masks
  --n-threads N_THREADS
                        Number of pool works to be used
  --remove              Remove previous images in the augmentation folders
```
### Training 
Training and model definition is performed in `trainVGG16unet.ipynb`. More information within that file.

### Validation
Validation is performed in `metrics.ipynb`. More information in that file.