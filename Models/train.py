import comet_ml
from baselines.intersection_based.inter_models import *
from baselines.utils.common_utils import seed_everything, get_device
from baselines.utils.inter_utils import *
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
device = get_device()
from comet_ml.integration.pytorch import log_model
from datetime import datetime
args = parse_args()

try:
    with open("api.key") as f:
        key = f.readline()[:-1]
except:
    raise FileNotFoundError("Comet key.api not found!")

experiment = comet_ml.Experiment(
    api_key=key,
    project_name="Parking_occupancy_SNP"
)

if args.name:
    experiment.set_name(args.name)

#Settings
min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
dataset = args.dataset

settings = {
    "batch_size" : args.batch,
    "epochs" : args.epoch,
    "learning_rate": args.rate,
    "dataframe" : "datasets/"+str(dataset)+"/"+str(dataset)+"/"+str(dataset)+"_dataframe.csv",
    "path" : "datasets/"+str(dataset)+'/'+str(dataset)+'/',
    "model_type" : args.model,
    "seed" : int(datetime.now().timestamp()),
    "save_rate" : args.saveRate,
    "pretrained" : args.pretrained
}
experiment.log_parameters(settings)
experiment.log_dataset_info(dataset, path = settings["path"])

seed_everything(settings["seed"])

#Get wanted model from inter models 
if settings["model_type"] == 'faster_rcnn_mobilenet':
    model = get_model(faster_rcnn_mobilenet_params, args.pretrained)
elif settings["model_type"] == 'faster_rcnn_mobilenetV3_Large':
    model = get_model(faster_rcnn_mobilenetV3_Large_params, args.pretrained)
elif settings["model_type"] == 'faster_rcnn_mobilenetV3_Small':
    model = get_model(faster_rcnn_mobilenetV3_Small_params, args.pretrained)
elif settings["model_type"] == 'faster_rcnn_resnet':
    model = get_model(faster_rcnn_resnet_params, args.pretrained)
elif settings["model_type"] == 'faster_rcnn_vgg':
    model = get_model(faster_rcnn_vgg_params, args.pretrained)
elif settings["model_type"] == 'retinanet_mobilenet':
    model = get_model(retinanet_mobilenet_params, args.pretrained)
elif settings["model_type"] == 'retinanet_resnet':
    model = get_model(retinanet_resnet_params, args.pretrained)
elif settings["model_type"] == 'retinanet_vgg':
    model = get_model(retinanet_vgg_params, args.pretrained)
else:
    raise Exception('Invalid model type')

#Loads exisint model for retraining
if args.saved:
    load_model(model, device, args.saved)

model.to(device)

DIR_INPUT = os.path.join(settings["path"], 'splitted_images/')
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_VAL = f'{DIR_INPUT}/val'
DIR_TEST = f'{DIR_INPUT}/test'

dataframe = pd.read_csv(settings["dataframe"])

train_df, valid_df = get_dataframes(dataframe)

# Dataset
train_dataset = ParkDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = ParkDataset(valid_df, DIR_VAL, get_valid_transform())

# Split the dataset in train and test set and create a data loader
indices = torch.randperm(len(train_dataset)).tolist()
train_data_loader = DataLoader(
    train_dataset,
    batch_size=settings["batch_size"],
    shuffle=True,
    num_workers=5,
    collate_fn=collate_fn
)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=settings["batch_size"],
    shuffle=False,
    num_workers=5,
    collate_fn=collate_fn
)

# Parameters that require grading (optimizing)
params = [p for p in model.parameters() if p.requires_grad]

#Create an optimizer
#SGD
#optimizer = torch.optim.SGD(params, lr=settings["learning_rate"], momentum=0.9, weight_decay=0.0005)
#Adam
optimizer = torch.optim.Adam(params, lr=settings["learning_rate"], weight_decay=0.001)

#lr_scheduler_increase = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)
lr_scheduler_decrease = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

train_inter_model(model, settings["epochs"], train_data_loader, valid_data_loader, device, experiment, settings, optimizer, scheduler=0, warmup=args.warmup)
#Save model to comet for inference
log_model(experiment, model, model_name=settings["model_type"])