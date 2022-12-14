import pandas as pd
import numpy as np
import cv2
import os
import re
import albumentations as A
import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision import models
import time
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser(description='Baseline training parameters')

    parser.add_argument('-d', '--dataframe', type=str, help='path to dataset dataframe')

    parser.add_argument('-p', '--path', type=int, help='path to the dataset')

    args = parser.parse_args()

    return args

args = parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if device.type == 'cpu':
    print('Using CPU')
else:
    print('Using GPU')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
num_classes = 2 

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


DIR_INPUT = os.path.join(args.path, 'splitted_images/')
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_VAL = f'{DIR_INPUT}/val'
DIR_TEST = f'{DIR_INPUT}/test'

train_df = pd.read_csv(args.dataframe)

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

train_part = train_df[train_df['folder'] == 'train']
val_part = train_df[train_df['folder'] == 'val']

valid_ids = val_part['image_id'].unique()
train_ids = train_part['image_id'].unique()

valid_df = val_part
train_df = train_part

class ParkDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(f"{self.image_dir}/{image_id}", cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
    
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

def get_train_transform():
    # do a number of image augmentations
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        MotionBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# model
model = model.to(device)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

# dataloaders
train_dataset = ParkDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = ParkDataset(valid_df, DIR_VAL, get_valid_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()
train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=4,
    collate_fn=collate_fn
)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
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


loss_hist = Averager()
loss_hist_val = Averager()
itr = 1
min_loss = -np.inf


for epoch in range(num_epochs):
    
    model.train()
    
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 5 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")
        itr += 1
        
    torch.save(model, os.path.join('Saved_Models/', dataset_name , group + "_" + str(epoch)+'.pth'))
    
    itr_val = 1
    
    
    with torch.no_grad():
        
        for val_images, val_targets, val_image_ids in valid_data_loader:
            
            val_images = list(val_image.to(device) for val_image in val_images)
            val_targets = [{val_k: val_v.to(device) for val_k, val_v in val_t.items()} for val_t in val_targets]

            val_loss_dict = model(val_images, val_targets)  

            val_losses = sum(val_loss for val_loss in val_loss_dict.values())
            val_loss_value = val_losses.item()

            loss_hist_val.send(val_loss_value)

            if itr_val % 5 == 0:
                print(f"Validation Batch loss: {val_loss_value}")
                
            itr_val += 1
            

    print(f"Epoch #{epoch} train loss: {loss_hist.value}")
    print(f"Epoch #{epoch} valid loss: {loss_hist_val.value}")    
    if loss_hist.value < min_loss:
        print(f"Epoch #{epoch} is best")
        min_loss = loss_hist.value


    