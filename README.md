# Contamination_DenseNet
Contamination classification for explants

This repository contains the pipeline for classifying explants into categories - contaminated, non-contaminated, missing

This is [Keras](https://keras.io/) implementation of DenseNet. The code for densenet used in this repository is obtained from [here](https://github.com/flyyufelix/cnn_finetune).

To know more about how DenseNet works, please refer to the [original paper](https://arxiv.org/abs/1608.06993)

```
Densely Connected Convolutional Networks
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
arXiv:1608.06993
```

### Setup environment.

1. Create a conda/python virtual environment.

2. Install the dependencies from requirements.txt using pip -

```
    pip install -r requirements.txt --no-cache-dir
```

### To run predictions on the dataset, please follow the below steps.

1. Create a directory for the input rgb images, and place the dataset into this directory.

   Please note that currently the code only supports grid type 12 (3 X 4).

2. Please place the downloaded model in the main working directory.

3. To run the script, use the command -

Please update the output directory path and image directory path in the config.py file (RGB_DIR and OUTPUT)

```
    KERAS_BACKEND=tensorflow python inference.py 
```

Or, Include img_dir and output_dir paths as arguments to the script.

```
    KERAS_BACKEND=tensorflow python inference.py --img_dir directory_path --output_file output_file_name 
```

Format of the output CSV - 

```
image_name,1,2,3,4,5,6,7,8,9,10,11,12
GWZ7_I2.0_F1.9_L80_194153_8_1_3_rgb.jpg,NC,NC,NC,NC,NC,NC,M,NC,NC,NC,NC,NC
GWZ7_I2.0_F1.9_L80_194927_1_2_5_rgb.jpg,C,NC,NC,NC,NC,NC,NC,NC,NC,NC,C,NC
GWZ7_I2.0_F1.9_L80_194429_11_2_2_rgb.jpg,NC,NC,NC,NC,NC,C,NC,NC,NC,NC,NC,NC
GWZ8_I2.0_F1.9_L80_195913_4_0_6_rgb.jpg,C,NC,NC,C,C,C,C,NC,C,C,C,NC
```

## Requirements

* Keras 2.0.5
