import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import pandas as pd

def process_dataset_splitted(dataset_path, add_source = False):
    
    JSONS = os.path.join(dataset_path, 'int_markup')
    IMAGES = os.path.join(dataset_path, 'images')
    
    
    TRAIN = os.path.join(dataset_path, 'splitted_images/train')
    VAL = os.path.join(dataset_path, 'splitted_images/val')
    TEST = os.path.join(dataset_path, 'splitted_images/test')
    
    
    train_list = os.listdir(TRAIN)
    val_list = os.listdir(VAL)
    test_list = os.listdir(TEST)
        
    
    image_list = os.listdir(IMAGES)
    json_list = os.listdir(JSONS)
    
    for i in image_list:
        if i.split('.')[-1] != 'jpg':
            image_list.remove(i)
            
    for i in json_list:
        if i.split('.')[-1] != 'json':
            json_list.remove(i)
            
    image_list = sorted(image_list)
    json_list = sorted(json_list)
    
    l = min(len(image_list), len(json_list))
    
    images = []
    bboxses = []
    folders = []
    folder = ''
    
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
        
        if dataset_name == 'ACMPS':
        
            bbxs = data
        
        else:
        
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
            
        
    if add_source:
        df = pd.DataFrame({'image_id': images,'bbox': bboxses, 'folder': folders, 'source': dataset_name})
    else:
        df = pd.DataFrame({'image_id': images,'bbox': bboxses, 'folder': folders})
                
    return df
