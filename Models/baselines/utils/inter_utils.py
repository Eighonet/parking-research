import numpy as np
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import re

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

def get_train_transform(add_augmentations=False, augmentation_list=[]):
    if add_augmentations:
        return A.Compose(augmentation_list, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


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

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def get_dataframes(original_dataframe):

    original_dataframe['x'] = -1
    original_dataframe['y'] = -1
    original_dataframe['w'] = -1
    original_dataframe['h'] = -1
    
    original_dataframe[['x', 'y', 'w', 'h']] = np.stack(original_dataframe['bbox'].apply(lambda x: expand_bbox(x)))
    original_dataframe.drop(columns=['bbox'], inplace=True)
    original_dataframe['x'] = original_dataframe['x'].astype(np.cfloat)
    original_dataframe['y'] = original_dataframe['y'].astype(np.cfloat)
    original_dataframe['w'] = original_dataframe['w'].astype(np.cfloat)
    original_dataframe['h'] = original_dataframe['h'].astype(np.cfloat)

    train_df = original_dataframe[original_dataframe['folder'] == 'train']
    valid_df = original_dataframe[original_dataframe['folder'] == 'val']

    return train_df, valid_df


def train_inter_model(model, num_epochs, train_data_loader, valid_data_loader, device):

    loss_hist = Averager()
    loss_hist_val = Averager()
    itr = 1
    min_loss = -np.inf
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

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
            
        torch.save(model, os.path.join('Saved_Models/', str(epoch)+'.pth'))
        
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