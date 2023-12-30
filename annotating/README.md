# Annotation and dataset creation instructions

The entire cycle of parking lot annotating consists of two stages:

- annotation of parking spaces (widget A).
- labelling of parking lot status (widget B).

##  Dataset preperation
Create the dataset directory into this format:
```
├── T10LOT
│   ├── images
└── T10LOT_annotation_classes
    ├── Distortion
    ├── Fog
    ├── Glare
    ├── Infrared
    ├── Night
    ├── Occlusion_car
    ├── Occlusion_tree
    ├── Overcast
    ├── Rainy
    ├── Shadow
    ├── Sunny
    └── Winter
```
Place all images into the images folder.
For better organization, use the `annotation_classes` folder and sort individual photos according to the folder names.
This is later used to create a json file with images labeled according to their class. This can be utilized in the future.

## Widget A usage
Set the datasets name as the function parameter and run the cell. A widget should appear that has the first photo from the dataset showing and a toolbar with buttons. Pressing the annotate and then clicking 3 point will create a box. Label every single parking place in this way. If all of the photos we taken by a stationary camera you can use the apply to all button which annotates all of the images with the currently made map. After clicking save a new `annotations.json` file shoudl appear in the root directory. Switch to the next widget.

## Widget B usage
Making sure that you have the `annotations.json` file generated enter the dataset name into the functions parameters and run the cell. Again you should see a widget open, now with the image having the boxes ploted on the image. By cliking on these boxes you label the parking slot as unoccupied (blue).
Do this for all of the images and then click the save and generate button. This should process the dataset into this structure:
```
├── T10LOT
│   ├── YOLO_COCO_markup
│   ├── images
│   ├── int_markup
│   ├── patch_markup
│   │   ├── Busy
│   │   └── Free
│   ├── patch_splitted
│   │   ├── test
│   │   │   ├── Busy
│   │   │   └── Free
│   │   ├── train
│   │   │   ├── Busy
│   │   │   └── Free
│   │   └── val
│   │       ├── Busy
│   │       └── Free
│   └── splitted_images
│       ├── test
│       ├── train
│       └── val
└── T10LOT_annotation_classes
    ├── Distortion
    ├── Fog
    ├── Glare
    ├── Infrared
    ├── Night
    ├── Occlusion_car
    ├── Occlusion_tree
    ├── Overcast
    ├── Rainy
    ├── Shadow
    ├── Sunny
    └── Winter
```
There should be `.json` or `.txt` files in the `markup` directories.
