import numpy as np
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import re
import matplotlib.pyplot as plt

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

def get_train_transform(add_augmentations=False, augmentation_list=[]):
    if add_augmentations:
        return A.Compose(augmentation_list, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    return A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

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


def train_inter_model(model, num_epochs, train_data_loader, valid_data_loader, device, experiment, settings, optimizer):

    save_epoch = 0
    loss_hist = Averager()
    loss_hist_val = Averager()
    min_loss = -np.inf
    #params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        itr = 1
        model.train()
        
        loss_hist.reset()
        
        for images, targets, image_ids in train_data_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            experiment.log_metric("training batch loss", loss_value)
            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            itr += 1
        
        itr_val = 1
        
        ##Validation
        with torch.no_grad():
            
            for val_images, val_targets, val_image_ids in valid_data_loader:
                # if itr_val == 1:
                #     for n, img in enumerate(val_images):
                #         experiment.log_image(img, name = "Epoch {}, image {} in valid batch {}".format(epoch, n, itr_val), annotations = val_targets[n])   
                # itr_val += 1
                
                val_images = list(val_image.to(device) for val_image in val_images)
                val_targets = [{val_k: val_v.to(device) for val_k, val_v in val_t.items()} for val_t in val_targets]

                val_loss_dict = model(val_images, val_targets)  

                val_losses = sum(val_loss for val_loss in val_loss_dict.values())
                val_loss_value = val_losses.item()
                
                experiment.log_metric("validation batch loss", val_loss_value)
                loss_hist_val.send(val_loss_value)
                itr_val = +1
                
        experiment.log_metric("epoch average loss", loss_hist.value, epoch = epoch)
        experiment.log_metric("epoch average validation loss", loss_hist_val.value, epoch = epoch)
        experiment.log_epoch_end(epoch)
        
        print(f"Epoch #{epoch} train loss: {loss_hist.value}")
        print(f"Epoch #{epoch} valid loss: {loss_hist_val.value}\n")  
          
        if loss_hist.value < min_loss:
            print(f"Epoch #{epoch} is best")
            min_loss = loss_hist.value
        
        #Save every 10 epochs localy
        if save_epoch == 10:
            if "Saved_Models" not in os.listdir():
                os.mkdir('Saved_Models')
            if settings["model_type"] not in os.listdir('Saved_Models/'):
                os.mkdir('Saved_Models/'+ settings["model_type"])
            torch.save(model.state_dict(), os.path.join('Saved_Models/'+settings["model_type"],'state_dict_'+str(epoch)+'.pth'))
            save_epoch = 0
        save_epoch +=1
        
    #Save after finishing training
    if "Saved_Models" not in os.listdir():
        os.mkdir('Saved_Models')
    if settings["model_type"] not in os.listdir('Saved_Models/'):
        os.mkdir('Saved_Models/'+ settings["model_type"])
    torch.save(model.state_dict(), os.path.join('Saved_Models/'+settings["model_type"],'state_dict_'+str(epoch)+'_final'+'.pth')) ##Final save


def show_from_dataset(n, train_data_loader):
    i = 0
    for images, targets, image_ids in train_data_loader:
        image = images
        target = targets
        image_id = image_ids
        if i == n:
            break
        i +=1
    pred_boxes = [[(x[0], x[1]), (x[2], x[3])] for x in list(targets[0]["boxes"].detach().numpy())]
    image = image[0].permute(1,2,0)
    image = image.detach().numpy()
    for x in pred_boxes:
        cv2.rectangle(image, (int(x[0][0]),int(x[0][1])), (int(x[1][0]),int(x[1][1])), color=(255, 0, 0), thickness=2)
    fig = plt.figure(figsize=(15,15))
    plt.imshow(image)
    plt.axis("off")
    
def make_pred(model, img_batch, treshold):
    pred = model(img_batch)
    pred_boxes = [[(x[0], x[1]), (x[2], x[3])] for x in list(pred[0]["boxes"].detach().numpy())]
    pred_class = list(pred[0]["labels"].detach().numpy())
    pred_score = list(pred[0]["scores"].detach().numpy())
    try:
        over_treshold = [pred_score.index(x) for x in pred_score if x>treshold][-1]
    except IndexError:
        raise ValueError("No detection above threshold")
    pred_boxes = pred_boxes[:over_treshold+1]
    pred_class = pred_class[:over_treshold+1]
    return pred_boxes, pred_score

def show_inference(img_batch, model, img, treshold):
    boxes, score = make_pred(model, img_batch, treshold)
    for i, x in enumerate(boxes):
        cv2.rectangle(img, (int(x[0][0]),int(x[0][1])), (int(x[1][0]),int(x[1][1])), color=(255, 0, 0), thickness=2)
        cv2.putText(img, str(score[i]), (int(x[0][0]),int(x[0][1])), cv2.LINE_AA, 1.2, (255,0,0), 1)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.axis("off")