import importlib  
Darknet = importlib.import_module("pytorch-YOLOv4.tool.darknet2pytorch.Darknet")
do_detect = importlib.import_module("pytorch-YOLOv4.tool.torch_utils.Darknet")
# from yolov4.tool.darknet2pytorch import Darknet
# from yolov4.tool.torch_utils import do_detect
import numpy as np
import cv2


def get_boxes_centroids(boxes, width, height):
    centroids = []
    for i in range(len(boxes)):
        box = boxes[i][:4]
        centroids.append([int(width * (box[0] + box[2]) / 2), int(height * (box[1] + box[3]) / 2)])

    return centroids


def get_spots_centroids(spots):
    centroids = []
    for spot in spots:
        x_spot = sum(spot[0]) // len(spot[0])
        y_spot = sum(spot[1]) // len(spot[1])
        centroids.append([x_spot, y_spot])

    return centroids


class YOLO_inf:
    def __init__(self, path_to_cfg, path_to_weights, inference=True):
        """
        ParkFinder YOLOv4 inference.
        """
        self.model = Darknet(path_to_cfg, inference).load_weights(path_to_weights)
    
    def _detect(self, image, th, nms_th):
        """
        Detect bounding boxes of cars.
          *  image  -- np.ndarray image of parking lot.
          *  th     -- threshold for object detection.
          *  nms_th -- non max suppression threshold. 
        """
        image = cv2.resize(image, (608, 608), interpolation = cv2.INTER_AREA)
        boxes = do_detect(model, image, th, nms_th, 0)[0]
        
        return boxes
    
    def predict(self, image, spots, th=0.3, nms_th=0.5):
        """
        Predict occupancy of parking slots.
          *  image  -- np.ndarray image of parking lot.
          *  spots  -- parking spots coordinates.
          *  th     -- threshold for object detection.
          *  nms_th -- non max suppression threshold. 
        """
        boxes = self._detect(image, th, nms_th)
        
        boxes_centroids = get_boxes_centroids(boxes, image.shape[1], image.shape[0])
        spots_centroids = get_spots_centroids(spots)
        spots_occup = {str(i+1):False for i in range(len(spots))}
        
        for i, box_centroid in enumerate(boxes_centroids):
            distances = []
            for j, spot_centroid in enumerate(spots_centroids):
                distances.append([math.hypot(spot_centroid[0]-box_centroid[0], spot_centroid[1]-box_centroid[1]), j])
            distances = sorted(distances, key=lambda x: x[0])
            
            spot_box = [boxes[i][0]*image.shape[1], boxes[i][1]*image.shape[0], boxes[i][2]*image.shape[1], boxes[i][3]*image.shape[0]]
            w = spot_box[2] - spot_box[0]
            h = spot_box[3] - spot_box[1]
            spot_th = w / 2 if w > h else h / 2
            if distances[0][0] < spot_th:
                spots_occup[str(distances[0][1]+1)] = True
            if distances[1][0] - distances[0][0] < 2:
                spots_occup[str(distances[1][1]+1)] = True
                
        return spots_occup
