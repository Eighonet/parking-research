import os
import json

def images_list_in_class(annotation_classes_list, main_folder):
    
    an_cl_path = f'{main_folder}_annotation_classes/'

    annotation_dict = {directory : os.listdir(f'{an_cl_path}{directory}') for directory in annotation_classes_list}
    images_list = os.listdir(main_folder)
    images_dict = {img: [] for img in images_list}

    for img in images_list:
        for directory in annotation_classes_list:
            if img in annotation_dict[directory]:
                images_dict[img].append(directory)

    with open(f'{main_folder}_image_level_markup.json', "w") as outfile:
        json.dump(images_dict, outfile)