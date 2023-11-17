# Revising deep learning methods in parking lot occupancy detection

Welcome to the research repo of the *''Revising deep learning methods in parking lot occupancy detection''* paper. Here we published the actual code regarding the parking lot occupancy detection problem considered in our study.

This is a fork for a personal study/research for a bachelor thesis.
Goal of this fork is to port the tools to newest python libraries and to serve as a personal playground.
Will propose fixes to the original repo, once done.

Anastasia Martynova, [Mikhail Kuznetsov](https://github.com/mmkuznecov), [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov), Vladislav Tishin, [Natalia Semenova](https://www.researchgate.net/profile/Natalia-Semenova-7).

arXiv PDF: https://arxiv.org/abs/2306.04288

# Datasets

In this section, we introduce the processed versions of datasets used in our experiments: ACPDS, ACMPS, CNRPark, PKLot and SPKL.

Links to the datasets:
- [ACPDS](https://sc.link/1KZq)
- [ACMPS](https://sc.link/1KZv)
- [PKLot](https://sc.link/1KZt)
- [CNRPark](https://sc.link/1KZr)
- [SPKLv2](https://sc.link/1KZu)

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

For each of the datasets, we also provided visual condition labels describing the presence of the special effects in the images:

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
- `Distortion` - parking lot is distorted or the image recorded by a camera with a wide angle.

Image-level annotations have the following structure:

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

Overall, each dataset's structure can be described as follows:

- `images` folder contains the images of parking lots;
- `int_markup` folder contains the intersection-level annotations of parking lots; 
- `patch_markup` folder contains the patch-level annotations of parking lots;
- `patch_splitted` folder contains the split patches of parking lots;
- `splitted_images` folder contains the split images of parking lots; 
- `DATASET_NAME_image_level_markup.json` file contains the image-level annotations of parking lots; 
- `busy_ilm_markup.json` and `free_ilm_markup.json` files contain the image-level annotations of free and occupied parking lots.

# Custom dataset

All scripts for data preprocessing are located in the `data_preprocessing` folder.

To preprocess your own dataset, you need to create a folder with the name of your dataset and put the images of parking lots in the `images` folder. After that, you should perform the labelling procedure using the widgets from the `annotators` folder. 

# Experiments running

In this study, we explored two approaches to the parking lot occupancy detection problem: patch-based classification and intersection-based classification.

List of implemented baseline models for patch-based classification:

CNN models:

- ResNet50;
- MobileNet;
- CarNet;
- AlexNet;
- mAlexNet;
- VGG-16;
- VGG-19;
- CFEN.

Vision transformer models:

- ViT (no prt);
- ViT (prt);
- DeiT (prt);
- PiT (prt);
- ViT (SPT LSA).

List of implemented baseline models for intersection-based classification:

- Faster R-CNN with ResNet-50 backbone;
- Faster R-CNN with MobileNet backbone;
- Faster R-CNN with VGG-19 backbone;
- RetinaNet with ResNet-50 backbone;
- RetinaNet with MobileNet backbone;
- RetinaNet with VGG-19 backbone.

All experiments were conducted on NVIDIA Tesla V100 GPUs.

# Prerequisites

To run the code, you need to install the requirements using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can create and activate the conda enviroment:

```bash
conda env create -f environment.yml
conda activate parking_research
```

We recommend using Python >= 3.7 and torchvision >= 1.7.0 as far as some of the architectures are not implemented in the previous versions.

# Baseline models

All baseline models are located in the `baselines` folder. All models are implemented using PyTorch and can be trained with the help of `train` files by passing the model name.

# Data processing

The data processing script `process_data.py` is located in the `data_preprocessing` folder. You should pass the name of your dataset as an argument:

```bash
python process_data.py --dataset DATASET_NAME
```

**Important!**: you should configure the following folder structure:

```
[DATASET_NAME]
│
└───[DATASET_NAME]
|
└───[DATASET_NAME]_annotation_classes
```

The structure of the included directories is described above in the `Data` section.

# Contact us

If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.

# License

Established code released as open-source software under the MIT license.

# Citation

To be updated.

```
@misc{martynova2023revising,
      title={Revising deep learning methods in parking lot occupancy detection}, 
      author={Anastasia Martynova and Mikhail Kuznetsov and Vadim Porvatov and Vladislav Tishin and Andrey Kuznetsov and Natalia Semenova and Ksenia Kuznetsova},
      year={2023},
      eprint={2306.04288},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
