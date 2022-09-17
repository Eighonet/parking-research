from tqdm import tqdm_notebook as tqdm
import json
import os

def get_patch_crops_ilm():
        
    with open(image_level_markup_json) as json_file:
        ilm = json.load(json_file) # ilm - image level markup
        
    busy_dict = {}
    free_dict = {}
    
    images = os.listdir(image_dir)
    count = 0
    a = 0
    
    for image_path in tqdm(images):
        if image_path.endswith('jpg'):

            annot_path = f"{annot_dir}/{image_path.split('.')[0]}.json"
            annot = json.load(open(annot_path))
            for lot in annot["lots"]:
                
                class_dir = "Busy" if lot['label'] else "Free"                
                crop_name = f"image_{count}.jpg"
                
                # we use try-except because some images have no annotations or some images are absent

                try:
                
                    if class_dir == "Busy":
                        busy_dict[crop_name] = ilm[image_path]
                    else:
                        free_dict[crop_name] = ilm[image_path]
                
                except:                    
                    print('Some lot markup is missing')                    

                count += 1

    # we have two files: busy and free
                
    with open("busy_ilm_markup.json", "w") as outfile:
        json.dump(busy_dict, outfile)
    
    with open("free_ilm_markup.json", "w") as outfile:
        json.dump(free_dict, outfile)