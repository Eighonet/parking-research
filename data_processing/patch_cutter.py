from tqdm import tqdm_notebook as tqdm
import numpy as np
import json
import cv2
import os


def patch_crop(image_dir, annot_dir, patch_dir):
    images = os.listdir(image_dir)

    os.system(f"mkdir {patch_dir}/Busy")
    os.system(f"mkdir {patch_dir}/Free")

    count = 0
    for image_path in tqdm(images):
        if image_path.split('.')[-1] == "jpg":
            annot_path = f"{annot_dir}/{image_path.split('.')[0]}.json"
            annot = json.load(open(annot_path))
            img = cv2.imread(f"{image_dir}/{image_path}")
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


