from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN, RetinaNet, faster_rcnn
import torchvision
import torch
import math


min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



#You can add custom models here, add them to the models list in inter_utils.py and in train.py for argument parser to work
faster_rcnn_mobilenet_params = {'backbone': 'mobilenet_v2', 
                                'out_channels': 1280,
                                'model': 'FasterRCNN'}

faster_rcnn_mobilenetV3_Large_params = {'backbone' : 'mobilenet_v3_large',
                                  'out_channels': 960,
                                  'model': 'FasterRCNN'}

faster_rcnn_mobilenetV3_Small_params = {'backbone' : 'mobilenet_v3_small',
                                  'out_channels': 576,
                                  'model': 'FasterRCNN'}

faster_rcnn_resnet_params = {'backbone': 'resnet50',
                                'out_channels': 2048,
                                'model': 'FasterRCNN'}

faster_rcnn_vgg_params = {'backbone': 'vgg19',
                          'out_channels': 512,
                          'model': 'FasterRCNN'}

retinanet_mobilenet_params = {'backbone': 'mobilenet_v2',
                              'out_channels': 1280,
                                'model': 'RetinaNet'}

retinanet_resnet_params = {'backbone': 'resnet50',
                           'out_channels': 2048,
                            'model': 'RetinaNet'}

retinanet_vgg_params = {'backbone': 'vgg19',
                        'out_channels': 512,
                        'model': 'RetinaNet'}

retinanet_mobilenetV3_Large_params = {'backbone' : 'mobilenet_v3_large',
                                  'out_channels': 960,
                                  'model': 'RetinaNet'}

retinanet_mobilenetV3_Small_params = {'backbone' : 'mobilenet_v3_small',
                                  'out_channels': 576,
                                  'model': 'RetinaNet'}


def get_model(model_params, pretrain = False):
    if not pretrain:
        if model_params['backbone'] == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2().features
        elif model_params['backbone'] == 'mobilenet_v3_small':
            model = torchvision.models.mobilenet_v3_small().features
        elif model_params['backbone'] == 'mobilenet_v3_large':
            model = torchvision.models.mobilenet_v3_large().features
        elif model_params['backbone'] == 'resnet50':
            model = torchvision.models.resnet50()
        elif model_params['backbone'] == 'vgg19':
            model = torchvision.models.vgg19().features
        
        if model_params['backbone'] == 'resnet50':
            modules = list(model.children())[:-1]
            backbone = torch.nn.Sequential(*modules)
        else:
            backbone = model
            
        backbone.out_channels = model_params['out_channels']
        new_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), 
                                            aspect_ratios=((0.5, 1.0, 2.0),))
        new_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=4, sampling_ratio=1)
        if model_params['model'] == 'FasterRCNN':
            model = FasterRCNN(backbone=backbone,
                                num_classes=2,
                                min_size=min_size,
                                max_size=max_size, 
                                image_mean=mean, 
                                image_std=std,
                                rpn_anchor_generator = new_anchor_generator,
                                box_roi_pool=new_roi_pooler)

        if model_params['model'] == 'RetinaNet':
            model = RetinaNet(backbone=backbone,
                                num_classes=2, 
                                min_size=min_size, 
                                max_size=max_size, 
                                image_mean=mean, 
                                image_std=std,
                                anchor_generator = new_anchor_generator)
    else:
        if model_params['model'] == 'FasterRCNN':
            if model_params['backbone'] == 'mobilenet_v3_large':
                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights = "DEFAULT")
            elif model_params['backbone'] == 'resnet50':
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = "DEFAULT")
            else:
                raise ValueError("No wanted pretrained model version found")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)
        elif model_params['model'] == 'RetinaNet':
            if model_params['backbone'] == 'resnet50':
                model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weight = "DEFAULT", num_classes = 2)
                # #Changing classification layer to classify 2 classes only
                # in_features = model.head.classification_head.conv[0].in_channels
                # num_anchors = model.head.classification_head.num_anchors
                # model.head.classification_head.num_classes = 2
                # cls_logits = torch.nn.Conv2d(2048, num_anchors * 2, kernel_size = 3, stride=1, padding=1)
                # #Pytorch documentation code
                # torch.nn.init.normal_(cls_logits.weight, std=0.01)
                # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
                # #Asssign the classification layer back to model
                # model.head.classification_head.cls_logits = cls_logits

            #raise ValueError("Pretrained for RetinaNet not implemented yet")
    return model