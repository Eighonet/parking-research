# YOLOv4 Inference
### Requirements
1. Install requirements:
```
numpy==1.21.5
opencv-python==4.2.0
```    
2. Clone this repo [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
3. Download weights:
```
sh download_weights.sh
```

### Usage
```python
model = YOLO_inf(PATH_TO_CFG, PATH_TO_WT)
m.predict(image, spots)
```    
Output:
```
{'1' : True,
 '2' : True,
 '3' : False,
    ...
 '17': True,
 '18': True,
 '19': True}
```
Visualization:
![Result](/images/result_detect.jpg)

