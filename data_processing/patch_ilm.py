from tqdm import tqdm_notebook as tqdm
import json
import cv2
import os

def patch_ilm(image_dir, annot_dir, image_level_markup_json):
    
    with open(image_level_markup_json) as json_file:
        ilm = json.load(json_file) # ilm - image level markup
        
    busy_dict = {}
    free_dict = {}
    
    images = os.listdir(image_dir)
    count = 0
    a = 0
    
    for image_path in tqdm(images):
        if image_path.split('.')[-1] == "jpg":
            annot_path = f"{annot_dir}/{image_path.split('.')[0]}.json"
            annot = json.load(open(annot_path))
            img = cv2.imread(f"{image_dir}/{image_path}")
            for lot in annot:
                
                class_dir = "Busy" if lot['label'] else "Free"                
                crop_name = f"image_{count}.jpg"
                
                try:
                
                    if class_dir == "Busy":
                        busy_dict[crop_name] = ilm[image_path]
                    else:
                        free_dict[crop_name] = ilm[image_path]
                
                except:                    
                    print('Some lot markup is missing')
                    a+=1
                    

                
                count += 1
                
    print(a)
    with open("busy_ilm_markup.json", "w") as outfile:
        json.dump(busy_dict, outfile)
    
    with open("free_ilm_markup.json", "w") as outfile:
        json.dump(free_dict, outfile)