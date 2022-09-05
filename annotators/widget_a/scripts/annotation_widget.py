from os import listdir
from os.path import isfile, join
import json

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ipywidgets import widgets, Dropdown, Box, Label, HBox, VBox

def b1():
    print('b1')

def pa_widget(markup_dir = "Downloads/parking", output_dir = ""):

    class Annotator(object):
        def __init__(self, axes):
            self.axes = axes

            self.xdata = []
            self.ydata = []
            
            self.annotation_dict = dict()
            self.image = ""

        def mouse_click(self, event):
            if not event.inaxes:
                return
            if annotatation_flag.value == '1':
                x, y = event.xdata, event.ydata

                self.xdata.append(x)
                self.ydata.append(y)
                line = Line2D([self.xdata[-2], self.xdata[-1]], [self.ydata[-2], self.ydata[-1]])
                line.set_color('r')
                self.axes.add_line(line)

                if len(self.xdata) % 4 == 0:
                    self.annotation_dict[self.image].append([self.xdata.copy(), self.ydata.copy()])
                    self.xdata.append(self.xdata[-4])
                    self.ydata.append(self.ydata[-4])

                    line = Line2D([self.xdata[-2], self.xdata[-1]], [self.ydata[-2], self.ydata[-1]])
                    line.set_color('r')
                    self.axes.add_line(line)
                    plt.draw()

                    self.xdata, self.ydata = [], []

    def forward_button_clicked(b):
        if 0 <= int(selected_image.value) < len(images) - 1:
            selected_image.value = str(int(selected_image.value) + 1)
            path.value = images[int(selected_image.value)]
            dropdown.value = path.value

            if path.value not in annotator.annotation_dict:
                annotator.annotation_dict[path.value], annotator.image = [], path.value
            else:
                annotator.image = path.value
            axes.clear()
            annotator.xdata, annotator.ydata = [], []
            for i in range(len(annotator.axes.lines)):
                annotator.axes.lines.pop(0)
            img = cv2.imread(markup_dir + '/' + path.value)
            if path.value not in metadata:
                metadata[path.value] = img.shape
                
            axes.imshow(img)
            plt.axis("off")

            shapes = annotator.annotation_dict[path.value]
            xdata, ydata = [], []
            for i in range(len(shapes)):
                xdata = shapes[i][0].copy()
                xdata.append(shapes[i][0][0])
                ydata = shapes[i][1].copy()
                ydata.append(shapes[i][1][0])
                for j in range(1, len(xdata)):
                    line = Line2D([xdata[j-1], xdata[j]], [ydata[j-1], ydata[j]])
                    line.set_color('r')
                    annotator.axes.add_line(line)

    def backward_button_clicked(b):
        if 1 <= int(selected_image.value) < len(images):
            selected_image.value = str(int(selected_image.value) - 1)
            path.value = images[int(selected_image.value)]
            dropdown.value = path.value

            if path.value not in annotator.annotation_dict:
                annotator.annotation_dict[path.value], annotator.image = [], path.value
            else:
                annotator.image = path.value
            axes.clear()
            annotator.xdata, annotator.ydata = [], []
            for i in range(len(annotator.axes.lines)):
                annotator.axes.lines.pop(0)
            img = cv2.imread(markup_dir + '/' + path.value)
            if path.value not in metadata:
                metadata[path.value] = img.shape
            
            axes.imshow(img)
            plt.axis("off")

            shapes = annotator.annotation_dict[path.value]
            xdata, ydata = [], []
            for i in range(len(shapes)):
                xdata = shapes[i][0].copy()
                xdata.append(shapes[i][0][0])
                ydata = shapes[i][1].copy()
                ydata.append(shapes[i][1][0])
                for j in range(1, len(xdata)):
                    line = Line2D([xdata[j-1], xdata[j]], [ydata[j-1], ydata[j]])
                    line.set_color('r')
                    annotator.axes.add_line(line)
                    
    def paint_button_clicked(b):
        if annotatation_flag.value == '0':
            annotatation_flag.value = '1'
            button_paint.style = {'button_color': 'red', 'color': 'white'}
        else:
            annotatation_flag.value = '0'
            button_paint.style = {'button_color': '#eeeeee'}

    def trash_button_clicked(b):
        annotator.xdata, annotator.ydata = [], []
        img = cv2.imread(markup_dir + '/' + path.value)
        if path.value not in metadata:
            metadata[path.value] = img.shape
            
        if len(annotator.axes.lines) > 0:
            if len(annotator.axes.lines) % 4 == 0:
                annotator.annotation_dict[annotator.image].pop(-1)
                for i in range(4):
                    annotator.axes.lines.pop(-1)
            else:
                while(len(annotator.axes.lines) % 4 != 0):
                    annotator.axes.lines.pop(-1)
            axes.imshow(img)

    def download_button_clicked(b):    
        with open(output_dir + "annotations.json", "w") as outfile:
            json.dump(annotator.annotation_dict, outfile)
       
        with open(output_dir + "metadata.json", "w") as outfile:
            json.dump(metadata, outfile)

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            path.value = change['new']
            selected_image.value = str(images_dict[path.value])
            if path.value not in annotator.annotation_dict:
                annotator.annotation_dict[path.value], annotator.image = [], path.value
            else:
                annotator.image = path.value
            axes.clear()
            annotator.xdata, annotator.ydata = [], []
            for i in range(len(annotator.axes.lines)):
                annotator.axes.lines.pop(0)
            img = cv2.imread(markup_dir + '/' + path.value)
            if path.value not in metadata:
                metadata[path.value] = img.shape
            
            axes.imshow(img)
            plt.axis("off")
            shapes = annotator.annotation_dict[path.value]
            xdata, ydata = [], []
            for i in range(len(shapes)):
                xdata = shapes[i][0].copy()
                xdata.append(shapes[i][0][0])
                ydata = shapes[i][1].copy()
                ydata.append(shapes[i][1][0])
                for j in range(1, len(xdata)):
                    line = Line2D([xdata[j-1], xdata[j]], [ydata[j-1], ydata[j]])
                    line.set_color('r')
                    annotator.axes.add_line(line)
               
    images = [f for f in listdir(markup_dir) if isfile(join(markup_dir, f)) and f[-4:] == ".jpg"]
    images_dict = {images[i]:i for i in range(len(images))}
    metadata = dict()

    selected_image = widgets.Label(value='0')
    path = widgets.Label(value=images[int(selected_image.value)])

    annotatation_flag = widgets.Label(value='0')

    img = cv2.imread(markup_dir + '/' + path.value)
    if path.value not in metadata:
        metadata[path.value] = img.shape 
        
    fig, axes = plt.subplots(figsize=(6, 6), num='Markup widget rev1')
    axes.imshow(img)
    plt.axis("off")

    annotator = Annotator(axes)
    annotator.annotation_dict[path.value], annotator.image = [], path.value

    plt.connect('button_press_event', annotator.mouse_click)

    button_paint = widgets.Button(description="🖌", style={'button_color': '#eeeeee'}, layout={'width': '35px'})
    button_trash = widgets.Button(description="🗑", layout={'width': '35px'})
    button_download = widgets.Button(description="⭳", layout={'width': '40px'})

    button_forward = widgets.Button(description="🠖", layout={'width': '35px'})
    button_backward = widgets.Button(description="🠔", layout={'width': '35px'})
    dropdown = Dropdown(options=images)
    dropdown_block = Box([Label(value='Select image'), dropdown])

    tool_box = HBox([button_paint, button_trash, button_backward, button_forward, dropdown, button_download])
    menu_box = VBox([tool_box])

    button_paint.on_click(paint_button_clicked)
    button_trash.on_click(trash_button_clicked)
    button_download.on_click(download_button_clicked)

    button_forward.on_click(forward_button_clicked)
    button_backward.on_click(backward_button_clicked)
    dropdown.observe(on_change)

    display(menu_box)