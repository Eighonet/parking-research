from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN, RetinaNet
import torchvision


min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

faster_rcnn_mobilenet_params = {'backbone': 'mobilenet_v2', 
                                'out_channels': 1280,
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


def get_model(model_params):

    if model_params['backbone'] == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    if model_params['backbone'] == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained=True).features
    if model_params['backbone'] == 'vgg19':
        backbone = torchvision.models.vgg19(pretrained=True).features

    backbone.out_channels = model_params['out_channels']

    new_anchor_generator = AnchorGenerator(sizes=((36, 64, 128, 256, 512),), 
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
    return model