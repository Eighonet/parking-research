# A comprehensive study on parking lot occupancy detection problem

Welcome to the main research repo of Parkfinder project. We present our intermediate results and publish the actual code regarding parking lot occupancy detection problem. 

Anastasia Martynova, Mikhail Kuznetsov, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov), Vladislav Tishin, [Natalia Semenova](https://www.researchgate.net/profile/Natalia-Semenova-7).


# Datasets

In this section we provide the links to the datasets used in our experiments. We used the following datasets: ACPDS, ACMPS, CNRPark, PKLot and self-collected dataset SPKL.

*Links for downloading the datasets will be provided soon.*

Dataset structure:

```
[DATASET_NAME]
│
└───images
│       image1.jpg
│       image2.jpg
│       ...
│
└───int_markup
│       image1.json
│       image2.json
│       ...
|
└───patch_markup
|       └───classes
|             └───Busy
|             |     image1.json
|             |     image2.json
|             |     ...
|             |
|             └───Free
|                   image3.json
|                   image4.json
|                   ...
|
│       image1.json
│       image2.json
|       image3.json
|       image4.json
│       ...
|
└───patch_splitted
|       └───train
|       |     └───Busy
|       |     |     patch1.jpg
|       |     |     patch2.jpg
|       |     |      ...
|       |     |
|       |     └───Free
|       |           patch3.jpg
|       |           patch4.jpg
|       |           ...
|       |
|       └───test
|       └───val
|
└───splitted_images
|       └───train
|       |     image1.jpg
|       |     image2.jpg
|       |     ...
|       |
|       └───test
|       └───val
|
[DATASET_NAME]_image_level_markup.json
busy_ilm_markup.json
free_ilm_markup.json
[DATASET_NAME]_dataframe.csv
```

In processed datasets we also provide image-level markup of parking lots. Image-level markup of parking lots describes if there is some kind of real-world occlusion on the image.

Overall we markup of 12 types image-level classes:

- `Sunny` - sunny weather;
- `Overcast` - overcast weather;
- `Rain` - rainy weather;
- `Winter` - winter weather, snow on the ground;
- `Fog` - foggy weather;
- `Glare` - glare on the image;
- `Night` - night time;
- `Infrared` - infrared image;
- `Occlusion (car)` - cars overlap each other;
- `Occlusion (tree)` - trees overlap cars and parking lots;
- `Distortion` - parking lot is distorted or its image made by a camera with a wide angle;
- `Shadow` - shadow on the image, parking lot is not so visible.

Image-level markup auxiliary dataset strutcture:

```
[DATASET_NAME]_annotation_classes
│
└───CLASS_NAME_1
│       image1.jpg
│       image2.jpg
│       ...
│
└───CLASS_NAME_2
│       image3.jpg
│       image4.jpg
│       ...
│
└───CLASS_NAME_3
│       image5.jpg
│       image6.jpg
│       ...
│
...
```

`images` folder contains the images of parking lots. `int_markup` folder contains the intersection-level markup of parking lots. `patch_markup` folder contains the patch-level markup of parking lots. `patch_splitted` folder contains the splitted patches of parking lots. `splitted_images` folder contains the splitted images of parking lots. `DATASET_NAME_image_level_markup.json` file contains the image-level markup of parking lots. `busy_ilm_markup.json` and `free_ilm_markup.json` files contain the image-level markup of busy and free parking lots.

# Custom dataset

All scripts for data preprocessing are located in the `data_preprocessing` folder.

To preprocess your own dataset you need to create a folder with the name of your dataset and put the images of parking lots in the `images` folder. After that you shoud label your dataset, using the widgets from `annotators` folder, instructions for them are located in the `annotators` folder too.

# Experiments running

In this research we propose two approaches to solve the parking lot occupancy detection problem: patch-based classification and intersection-based classification.

List of implemented baseline models for patch-based classification:

Convolution-based models:

- ResNet50;
- MobileNet;
- CarNet;
- AlexNet;
- mAlexNet;
- VGG-16;
- VGG-19;
- CFEN;

Transformer-based models:

- ViT (no prt);
- ViT (prt);
- DeiT (prt);
- PiT (prt);
- ViT (SPT LSA);

List of implemented baseline models for detection-based classification:

- Faster R-CNN with ResNet-50 backbone;
- Faster R-CNN with MobileNet backbone;
- Faster R-CNN with VGG-19 backbone;
- RetinaNet with ResNet-50 backbone;
- RetinaNet with MobileNet backbone;
- RetinaNet with VGG-19 backbone;

All experiments were run on NVIDIA Tesla V100 GPUs.

# Prerequisites

To run the code you can install the requirements using the following command:

```bash
pip install -r requirements.txt
```

Or you can create and activate conda enviroment

```bash
conda env create -f environment.yml
conda activate parking_research
```

We recommend to use Python version >= 3.7. We also recommend to use GPU for training and inference. Also we recommend to use torchvision >= 1.7.0 as far as some of the architectures are not implemented in the previous versions.

# Baseline models

All baseline models are located in the `baselines` folder. All models are implemented using PyTorch. You can train models using `train` files by just passing model name respectively.

# Contact us

If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.

# License

Established code released as open-source software under the MIT license.
