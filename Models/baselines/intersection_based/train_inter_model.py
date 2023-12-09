import sys
sys.path.append(r'C:\Users\slava\Documents\AI\parking-research-argon')
from inter_models import *
from baselines.utils.common_utils import seed_everything, get_device
from baselines.utils.inter_utils import *
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset

seed_everything(seed=42)
device = get_device()
args = parse_args()

if args.model_type == 'faster_rcnn_mobilenet':
    model = get_model(faster_rcnn_mobilenet_params)
elif args.model_type == 'faster_rcnn_resnet':
    model = get_model(faster_rcnn_resnet_params)
elif args.model_type == 'faster_rcnn_vgg':
    model = get_model(faster_rcnn_vgg_params)
elif args.model_type == 'retinanet_mobilenet':
    model = get_model(retinanet_mobilenet_params)
elif args.model_type == 'retinanet_resnet':
    model = get_model(retinanet_resnet_params)
elif args.model_type == 'retinanet_vgg':
    model = get_model(retinanet_vgg_params)
else:
    raise Exception('Invalid model type')

model.to(device)


DIR_INPUT = os.path.join(args.path, 'splitted_images/')
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_VAL = f'{DIR_INPUT}/val'
DIR_TEST = f'{DIR_INPUT}/test'

dataframe = pd.read_csv(args.dataframe)

train_df, valid_df = get_dataframes(dataframe)

# dataloaders
train_dataset = ParkDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = ParkDataset(valid_df, DIR_VAL, get_valid_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()
train_data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    # num_workers=4,
    collate_fn=collate_fn
)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    # num_workers=4,
    collate_fn=collate_fn
)

images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler_increase = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)
lr_scheduler_decrease = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 35

train_inter_model(model, num_epochs, train_data_loader, valid_data_loader)