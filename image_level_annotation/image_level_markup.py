import os
import json

'''
You should have following directory stucture: original dataset with images and annotation classes folders
'''

# ilm = image level markup

def get_ilm_json(annotation_classes_list, main_folder):

    annotation_dict = {directory : os.listdir(directory) for directory in annotation_classes_list}
    images_list = os.listdir(main_folder)
    images_dict = {img: [] for img in images_list}

    for img in images_list:
        for directory in annotation_classes_list:
            if img in annotation_dict[directory]:
                images_dict[img].append(directory)

    with open(f'{main_folder}_image_level_markup.json', "w") as outfile:
        json.dump(images_dict, outfile)


if __name__ == '__main__':

    annotation_classes_list = ['Snowfall', 'Smoke_Fog', 'Sunny', 'Night', 
                    'Occlusion_tree', 'Shadow', 'Snow', 'Perspective_distortion', 
                    'Rainy', 'Glare', 'Occlusion_car', 'Overcast', 'Infrared']
    
    main_folder = 'CNRParkEXT' # replace with your main folder name (ACPDS, CNRParkEXT, etc.)
    get_ilm_json(annotation_classes_list, main_folder)

