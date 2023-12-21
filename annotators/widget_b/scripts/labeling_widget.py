from os import listdir
from os.path import isfile, join
import json
import re

from IPython.display import display, HTML, Image
from ipywidgets import widgets, Dropdown, Box, Label, HBox, VBox, interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import *
from ipyevents import Event 

import pandas as pd
pd.set_option('display.max_colwidth', None)

from io import BytesIO
import warnings
import string
import numpy as np

warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = {1: (0, 0, 255), 0: (255, 0, 0)} 
thickness = 4
isClosed = True

def pm_widget(annotation_path: str="annotations.json", image_dirs : str = "img", output_dir:str=""):
    class Annotator(object):
        def __init__(self):
            #Loading annotations from dict to touple cords
            def to_tuple(markup: list) -> list:
                for i in range(len(markup)):
                    markup[i] = (tuple(markup[i][0]),tuple(markup[i][1]))
                return markup
            
            #Opening JSON reading and loadig into dict markup
            with open(annotation_path) as f:
                self.markup = json.load(f)
                 
            img_dict_names = sorted(list(self.markup.keys()))
            self.labels = {}
            for i in img_dict_names:
                data = to_tuple(self.markup[i])
            self.labels = {file:{data[j]:1 for j in range(len(data))} for file in img_dict_names}
            self.current_image = img_dict_names[0]
            
    def get_image(path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    def draw_lines(byte_image: bytes) -> bytes:
        current_markup = annotator.labels[annotator.current_image]
        decoded = cv2.imdecode(np.frombuffer(byte_image, np.uint8), -1)
        for coords in current_markup:
            lot_markup = np.array(coords).T.astype("int")
            color = colors[current_markup[coords]]
            cv2.polylines(decoded, [lot_markup], isClosed, color, thickness=thickness)
        decoded_bytes = cv2.imencode('.jpg', decoded)[1].tobytes()
        return decoded_bytes
    
    # Was for changing folder useless now          
    # def on_change(change) -> None:
    #     if change["type"] == "change" and change["name"] == "value":
    #         annotator.current_folder = str(change["new"]) + ".jpg"
    #         annotator.current_image = [file for file in listdir(str(change["new"]))][0]
    #         selected_folder.value = str(change["new"])
            
    #         corresponding_images.value = str([f for f in listdir(str(change["new"])) if f[0] != "."])
    #         processed_names = process_list(corresponding_images.value)
    #         image_dropdown.options = processed_names
            
    #         byte_image = get_image(selected_folder.value + "/" + processed_names[0])    
    #         image.value = draw_lines(byte_image)
            
            
    def on_change_image(change) -> None:
        if change["type"] == "change" and change["name"] == "value":
            annotator.current_image = str(change["new"])
            byte_image = get_image(image_dirs + "/" + str(change["new"]))
            image.value = draw_lines(byte_image)

    def download_button_clicked(b, output_dir:str=""):
        
        try:
            os.mkdir(output_dir + "int_markup")
            os.mkdir(output_dir + "patch_markup")
        except:
            pass
        
        for folder in annotator.labels.keys():
                current_markup = annotator.labels[folder]
                image_rois = current_markup
                
                standard_output = {"lots":[]}
                patch_output = {"lots":[]}

                for place in image_rois.keys():
                    points = np.array(place).T.astype("int")
                    lot = {"coordinates":points.tolist(), "label": int(image_rois[place])}

                    x_max, x_min = int(max(np.array(points)[:, 0])), int(min(np.array(points)[:, 0]))
                    y_max, y_min = int(max(np.array(points)[:, 1])), int(min(np.array(points)[:, 1]))
                    lot_patch = {"coordinates":[[x_max, y_max], [x_max, y_min], 
                                                [x_min, y_min], [x_min, y_max]], "label": image_rois[place]}
                    standard_output["lots"].append(lot)
                    patch_output["lots"].append(lot_patch)

                with open(output_dir + "int_markup/" +  folder[:-4] + '.json', 'w') as f:
                    json.dump(standard_output, f)

                with open(output_dir + "patch_markup/" + folder[:-4] + '.json', 'w') as f:
                    json.dump(patch_output, f)     

    def process_list(input_str: str) -> list:
        return re.sub(r'[\'\[\]]', ' ', input_str).replace(" ", "").split(",")

    def update_coords(event):
        coordinates.value = str([event['dataX'], event['dataY']])
        current_markup = annotator.labels[annotator.current_image]
        for coords in current_markup:
            points = np.array(coords).T.astype("int")
            check = cv2.pointPolygonTest(points, (event['dataX'], event['dataY']), False)
            if check == 1:
                annotator.labels[annotator.current_image][coords] =\
                int(not annotator.labels[annotator.current_image][coords])
        byte_image = get_image(image_dirs + "/" + annotator.current_image)
        image.value = draw_lines(byte_image)

    def forward_button_clicked(b):
        folders = sorted(list(annotator.markup.keys()))
        files = [file for file in listdir(annotator.current_image[:-4])]
        idx = files.index(annotator.current_image)
        if idx < len(files) - 1:
            image_dropdown.value = files[idx+1]
            annotator.current_image = files[idx+1]
            byte_image = get_image(image_dirs + "/" + str(files[idx+1]))
            image.value = draw_lines(byte_image)


    def backward_button_clicked(b):
        folders = sorted(list(annotator.markup.keys()))
        files = [file for file in listdir(annotator.current_image[:-4])]
        idx = files.index(annotator.current_image)
        if idx > 0:
            image_dropdown.value = files[idx-1]
            annotator.current_image = files[idx-1]
            byte_image = get_image(image_dirs.value + "/" + str(files[idx-1]))
            image.value = draw_lines(byte_image)


    annotator = Annotator()
    
    # dropdowns
    corresponding_images = widgets.Label(value=str([f for f in listdir(image_dirs) if f[0] != "."]))
    processed_names = process_list(corresponding_images.value)
    image_dropdown = widgets.Dropdown(options=processed_names)

    #buttons
    button_forward = widgets.Button(description="→", layout={"width": "35px"})
    button_backward = widgets.Button(description="←", layout={"width": "35px"})
    button_download = widgets.Button(description="Save", layout={"width": "60px"})

    coordinates = HTML('[]')

    # image block init
    byte_image = get_image(image_dirs + "/" + processed_names[0])
    image = widgets.Image(
    value=draw_lines(byte_image),
    format='jpg'
    )

    # tab init
    tab_nest = widgets.Tab()

    # control panel
    panel = HBox([image_dropdown, button_backward, button_forward, button_download])

    # tab filled
    tab_nest.children = [VBox(children = (image, panel))]

    tab_nest.set_title(0, 'Markup v2.1')

    # handlers
    image_dropdown.observe(on_change_image)

    im_events = Event()
    im_events.source = image
    im_events.watched_events = ['click']
    im_events.on_dom_event(update_coords)
    no_drag = Event(source=image, watched_events=['dragstart'], prevent_default_action = True)

    button_download.on_click(download_button_clicked)
    button_forward.on_click(forward_button_clicked)
    button_backward.on_click(backward_button_clicked)

    # visual
    display(tab_nest)