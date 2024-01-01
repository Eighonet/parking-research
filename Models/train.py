import comet_ml
import inquirer
from baselines.utils.queries import *
from baselines.intersection_based.inter_models import *
from baselines.utils.common_utils import seed_everything, get_device
from baselines.utils.inter_utils import *
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from comet_ml.integration.pytorch import log_model
from datetime import datetime

try:
    with open("api.key") as f:
        api_key = f.readline()[:-1]
except:
    raise FileNotFoundError("Comet key.api not found!")
datasets = []
for dataset_name in os.listdir("datasets"):
    if os.path.isdir(os.path.join("datasets",dataset_name)):
        datasets.append(dataset_name)
if not datasets:
    raise FileNotFoundError("No dataset found")

answers = inquirer.prompt(questions, raise_keyboard_interrupt=True)
datasets_questions = []
for n in range(int(answers["dataset_n"])):
    datasets_questions.extend([inquirer.List(f'datasert_{n}', message=f'Choose dataset {n}', choices= datasets),
                              inquirer.Text(f'epoch_{n}', message=f'Number of epoochs?', validate = check_int,  default = 50)])
datasets_epochs = inquirer.prompt(datasets_questions, raise_keyboard_interrupt=True)

datasets = []
epochs = []
for n, (key, item) in enumerate(datasets_epochs.items()):
    if n % 2:
        epochs.append(item)
    else:
        datasets.append(item)

device = get_device()
#Settings
min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

settings = {
    "batch_size" : int(answers["batch"]),
    "learning_rate": float(answers["rate"]),
    "model_type" : answers["model"],
    "seed" : int(datetime.now().timestamp()),
    "save_rate" : int(answers["save_rate"]),
    "pretrained" : answers["pretrained"],
    "dataframe" : "datasets/"+datasets[0]+"/"+datasets[0]+"/"+datasets[0]+"_dataframe.csv",
    "path": "datasets/"+datasets[0]+'/'+datasets[0]+'/',
    "epochs" : int(epochs[0])
}

seed_everything(settings["seed"])

#Get wanted model from inter models, add custom models here too
if settings["model_type"] == 'faster_rcnn_mobilenet':
    model = get_model(faster_rcnn_mobilenet_params, answers["pretrained"])
elif settings["model_type"] == 'faster_rcnn_mobilenetV3_Large':
    model = get_model(faster_rcnn_mobilenetV3_Large_params, answers["pretrained"])
elif settings["model_type"] == 'faster_rcnn_mobilenetV3_Small':
    model = get_model(faster_rcnn_mobilenetV3_Small_params, answers["pretrained"])
elif settings["model_type"] == 'faster_rcnn_resnet':
    model = get_model(faster_rcnn_resnet_params, answers["pretrained"])
elif settings["model_type"] == 'faster_rcnn_vgg':
    model = get_model(faster_rcnn_vgg_params, answers["pretrained"])
elif settings["model_type"] == 'retinanet_mobilenet':
    model = get_model(retinanet_mobilenet_params, answers["pretrained"])
elif settings["model_type"] == 'retinanet_resnet':
    model = get_model(retinanet_resnet_params, answers["pretrained"])
elif settings["model_type"] == 'retinanet_vgg':
    model = get_model(retinanet_vgg_params, answers["pretrained"])
elif settings["model_type"] == 'retinanet_mobilenetV3_Large':
    model = get_model(retinanet_mobilenetV3_Large_params, answers["pretrained"])
elif settings["model_type"] == 'retinanet_mobilenetV3_Small':
    model = get_model(retinanet_mobilenetV3_Small_params, answers["pretrained"])
else:
    raise Exception('Invalid model type')

#Loads exisint model for retraining
if answers["retrain"]:
    load_model(model, device, answers["saved"])

model.to(device)

# Parameters that require grading (optimizing)
params = [p for p in model.parameters() if p.requires_grad]
    
#Trains for multiple datasets
for i, dataset in enumerate(datasets):
    print("Training on", dataset)
    settings["dataframe"] = "datasets/"+dataset+"/"+dataset+"/"+dataset+"_dataframe.csv"
    settings["path"] = "datasets/"+dataset+'/'+dataset+'/'
    settings["epochs"] = int(epochs[i])

    experiment = comet_ml.Experiment(
    api_key=api_key,
    project_name="Parking_occupancy"
    )

    experiment.set_name(answers["name"]+"_"+dataset)
    experiment.log_parameters(settings)
    experiment.log_dataset_info(dataset, path = settings["path"])

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
        num_workers=4,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    #Create an optimizer
    #SGD
    #optimizer = torch.optim.SGD(params, lr=settings["learning_rate"], momentum=0.9, weight_decay=0.0005)
    #Adam
    optimizer = torch.optim.Adam(params, lr=settings["learning_rate"], weight_decay=0.001)

    #lr_scheduler_increase = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)
    lr_scheduler_decrease = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_inter_model(model, settings["epochs"], train_data_loader, valid_data_loader, device, experiment, settings, optimizer, scheduler=0, warmup=answers["warmup"])
    
    if not i == len(datasets):
        experiment.end()

    #Save model to comet for inference
    log_model(experiment, model, model_name=settings["model_type"])