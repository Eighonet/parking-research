import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import random
import pandas as pd

def img_annotation_classes(dataset):

    annot_classes_dir = f"{dataset}/{dataset}_annotation_classes"
    main_dir = f"{dataset}/{dataset}"

    full_image_set = set(os.listdir(f'{main_dir}/images'))

    annotation_classes = os.listdir(annot_classes_dir)
    annotation_sets = [set(os.listdir(f'{annot_classes_dir}/{annot_class}')) for annot_class in annotation_classes]

    annot_classes_dict = {img: [] for img in full_image_set}

    for annot_class, annot_set in zip(annotation_classes, annotation_sets):
        for img in annot_set:
            annot_classes_dict[img].append(annot_class)

    with open(f'{dataset}/{dataset}/{dataset}_image_level_markup.json', "w") as outfile:
        json.dump(annot_classes_dict, outfile)


def patch_crop(dataset):

    main_dir = f"{dataset}/{dataset}"
    images_dir = f"{main_dir}/images"
    int_markup_dir = f"{main_dir}/int_markup"
    patch_dir = f"{main_dir}/patch_markup"

    os.mkdir(patch_dir)
    os.mkdir(f"{patch_dir}/Busy")
    os.mkdir(f"{patch_dir}/Free")

    images = os.listdir(images_dir)

    count = 0
    for image_path in tqdm(images):
        if image_path.split('.')[-1] == "jpg":
            annot_path = f"{int_markup_dir}/{image_path.split('.')[0]}.json"
            annot = json.load(open(annot_path))
            img = cv2.imread(f"{images_dir}/{image_path}")
            for lot in annot["lots"]:
                np_cors = np.array(lot['coordinates']).T
                x_min = int(np_cors[0].min())
                x_max = int(np_cors[0].max())
                y_min = int(np_cors[1].min())
                y_max = int(np_cors[1].max())
                cropped_image = img[y_min:y_max, x_min:x_max]
                class_dir = "Busy" if lot['label'] else "Free"
                cv2.imwrite(f"{patch_dir}/{class_dir}/image_{count}.jpg", cropped_image)
                count += 1


def patch_ilm(dataset):

    main_dir = f"{dataset}/{dataset}"
    images_dir = f"{main_dir}/images"
    int_markup_dir = f"{main_dir}/int_markup"


    image_level_markup_json = f"{dataset}/{dataset}/{dataset}_image_level_markup.json"

    image_level_markup = json.load(open(image_level_markup_json))

    busy_dict = {}
    free_dict = {}
    
    images = os.listdir(images_dir)
    count = 0
    error_count = 0

    for image_path in tqdm(images):
        if image_path.split('.')[-1] == "jpg":
            annot_path = f"{int_markup_dir}/{image_path.split('.')[0]}.json"
            annot = json.load(open(annot_path))
            for lot in annot["lots"]:
                
                class_dir = "Busy" if lot['label'] else "Free"                
                crop_name = f"image_{count}.jpg"
                
                try:
                    if class_dir == "Busy":
                        busy_dict[crop_name] = image_level_markup[image_path]
                    else:
                        free_dict[crop_name] = image_level_markup[image_path]
                except:                    
                    error_count+=1
                    

                
                count += 1


    print(f"Error count: {error_count}")

    with open(f'{dataset}/{dataset}/{dataset}_busy_patch_markup.json', "w") as outfile:
        json.dump(busy_dict, outfile)

    with open(f'{dataset}/{dataset}/{dataset}_free_patch_markup.json', "w") as outfile:
        json.dump(free_dict, outfile)

def splitter(dataset, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3):

    assert train_ratio + val_ratio + test_ratio == 1, "Train, val and test ratios must sum to 1"

    main_dir = f"{dataset}/{dataset}"
    patch_dir = f"{main_dir}/patch_markup"
    int_markup_dir = f"{main_dir}/int_markup"

    os.mkdir(f"{main_dir}/splitted_images")
    os.mkdir(f"{main_dir}/splitted_images/train")
    os.mkdir(f"{main_dir}/splitted_images/val")
    os.mkdir(f"{main_dir}/splitted_images/test")

    os.mkdir(f"{main_dir}/patch_splitted")
    os.mkdir(f"{main_dir}/patch_splitted/train")
    os.mkdir(f"{main_dir}/patch_splitted/val")
    os.mkdir(f"{main_dir}/patch_splitted/test")

    os.mkdir(f"{main_dir}/patch_splitted/train/Busy")
    os.mkdir(f"{main_dir}/patch_splitted/train/Free")
    os.mkdir(f"{main_dir}/patch_splitted/val/Busy")
    os.mkdir(f"{main_dir}/patch_splitted/val/Free")
    os.mkdir(f"{main_dir}/patch_splitted/test/Busy")
    os.mkdir(f"{main_dir}/patch_splitted/test/Free")

    busy_images = os.listdir(f"{patch_dir}/Busy")
    free_images = os.listdir(f"{patch_dir}/Free")

    # shuffle images

    random.shuffle(busy_images)
    random.shuffle(free_images)

    # split images

    busy_train = busy_images[:int(len(busy_images)*train_ratio)]
    busy_val = busy_images[int(len(busy_images)*train_ratio):int(len(busy_images)*(train_ratio+val_ratio))]
    busy_test = busy_images[int(len(busy_images)*(train_ratio+val_ratio)):]

    free_train = free_images[:int(len(free_images)*train_ratio)]
    free_val = free_images[int(len(free_images)*train_ratio):int(len(free_images)*(train_ratio+val_ratio))]
    free_test = free_images[int(len(free_images)*(train_ratio+val_ratio)):]

    # move images

    for image in busy_train:
        shutil.copy(f"{patch_dir}/Busy/{image}", f"{main_dir}/patch_splitted/train/Busy/{image}")

    for image in busy_val:
        shutil.copy(f"{patch_dir}/Busy/{image}", f"{main_dir}/patch_splitted/val/Busy/{image}")

    for image in busy_test:
        shutil.copy(f"{patch_dir}/Busy/{image}", f"{main_dir}/patch_splitted/test/Busy/{image}")

    for image in free_train:
        shutil.copy(f"{patch_dir}/Free/{image}", f"{main_dir}/patch_splitted/train/Free/{image}")

    for image in free_val:
        shutil.copy(f"{patch_dir}/Free/{image}", f"{main_dir}/patch_splitted/val/Free/{image}")

    for image in free_test:
        shutil.copy(f"{patch_dir}/Free/{image}", f"{main_dir}/patch_splitted/test/Free/{image}")

    images_list = os.listdir(f"{main_dir}/images")
    random.shuffle(images_list)
    train_images = images_list[:int(len(images_list)*train_ratio)]
    val_images = images_list[int(len(images_list)*train_ratio):int(len(images_list)*(train_ratio+val_ratio))]
    test_images = images_list[int(len(images_list)*(train_ratio+val_ratio)):]

    for image in train_images:
        shutil.copy(f"{main_dir}/images/{image}", f"{main_dir}/splitted_images/train/{image}")

    for image in val_images:
        shutil.copy(f"{main_dir}/images/{image}", f"{main_dir}/splitted_images/val/{image}")

    for image in test_images:
        shutil.copy(f"{main_dir}/images/{image}", f"{main_dir}/splitted_images/test/{image}")

    print("Splitting done!")

def get_dataframe(dataset):

    main_dir = f"{dataset}/{dataset}"

    JSONS = f'{main_dir}/int_markup'
    IMAGES = f'{main_dir}/images'

    TRAIN = f'{main_dir}/splitted_images/train'
    VAL = f'{main_dir}/splitted_images/val'
    TEST = f'{main_dir}/splitted_images/test'

    train_list = os.listdir(TRAIN)
    val_list = os.listdir(VAL)
    test_list = os.listdir(TEST)

    image_list = os.listdir(IMAGES)
    json_list = os.listdir(JSONS)

    image_list = sorted(image_list)
    json_list = sorted(json_list)

    # in order to pre
    l = min(len(image_list), len(json_list))

    images = []
    bboxses = []
    folders = []

    for i in range(l):

        json_path = os.path.join(JSONS, json_list[i])
        image_name = image_list[i]

        if image_name in train_list:
            folder = 'train'
        if image_name in val_list:
            folder = 'val'
        if image_name in test_list:
            folder = 'test'

        f = open(json_path)
        data = json.load(f)
        f.close()

        bbxs = data['lots']

        for j in range(len(bbxs)):

            if bbxs[j]['label'] == 1:
                images.append(image_name)
                folders.append(folder)

                y_s = [bbxs[j]['coordinates'][0][1], bbxs[j]['coordinates'][1][1], bbxs[j]['coordinates'][2][1], bbxs[j]['coordinates'][3][1]]
                x_s = [bbxs[j]['coordinates'][0][0], bbxs[j]['coordinates'][1][0], bbxs[j]['coordinates'][2][0], bbxs[j]['coordinates'][3][0]]

                y_max = max(y_s)
                y_min = min(y_s)
                x_max = max(x_s)
                x_min = min(x_s)

                w = x_max - x_min
                h = y_max - y_min

                bb = [x_min, y_min, w, h]
                bboxses.append(bb)


        df = pd.DataFrame({'image_id': images,'bbox': bboxses, 'folder': folders})
        df.to_csv(f'{main_dir}/{dataset}_dataframe.csv', index=False)


def main():

    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('--dataset', type=str, help='Path to dataset')

    args = parser.parse_args()
    dataset = args.dataset

    #img_annotation_classes(dataset)
    #patch_crop(dataset)
    #patch_ilm(dataset)
    #splitter(dataset)
    get_dataframe(dataset)

if __name__ == '__main__':
    main()