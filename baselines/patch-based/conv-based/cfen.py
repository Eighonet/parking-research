import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
import torchvision.utils
import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import warnings
import time
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, roc_auc_score

NUM_WORKERS = 4
SIZE_H = 144
SIZE_W = 96
BATCH_SIZE = 64
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

def parse_args():

    parser = argparse.ArgumentParser(description='Baseline training parameters')

    parser.add_argument('-p', '--path', type=str, help='path to the dataset')

    args = parser.parse_args()

    return args

args = parse_args()

class STN(nn.Module):
    def __init__(self, in_channels, h, w, use_dropout=False):
        super(STN, self).__init__()
        self._h = h
        self._w = w
        self._in_ch = in_channels 
        self.dropout = use_dropout

        # localization net 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size= 5, stride=1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size= 5, stride=1, padding = 0)

        self.fc1 = nn.Linear(33*21*20, 20)
        self.fc2 = nn.Linear(20, 3 * 2)


    def forward(self, x): 
        
        batch_images = x
        x = F.relu(self.conv1(x.detach()))  #140x92x20
        x = F.max_pool2d(x, 2) #70x46x20
        x = F.relu(self.conv2(x))  #66x42x20
        x = F.max_pool2d(x,2) #33x21x20
        x = x.view(-1, 33*21*20)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)
        
        x = x.view(-1, 2,3)
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0))
        r = F.grid_sample(batch_images, affine_grid_points)
        
        return r, affine_grid_points


class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform):
        
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        label = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
        return img0, img1, label, img0_tuple[1], img1_tuple[1]
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


train_dataset = torchvision.datasets.ImageFolder(f'{args.path}/patch_splitted/train')
valid_dataset = torchvision.datasets.ImageFolder(f'{args.path}/patch_splitted/val')
train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)

transformer = torchvision.transforms.Compose([
    transforms.Resize((SIZE_H, SIZE_W)),
    transforms.ToTensor(), 
    transforms.Normalize(image_mean, image_std)
])

siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=train_dataset, transform=transformer)
siamese_dataset_valid = SiameseNetworkDataset(imageFolderDataset=valid_dataset, transform=transformer)


class CFEN(nn.Module):

    def __init__(self):
        super(CFEN, self).__init__()
        self.stnmod = STN(in_channels=3, h=144 ,w=96)

        # CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size= 1, stride=1, padding = 0), #144x96x16
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size= 1, stride=1, padding = 0), #144x96x32
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 1, stride=1, padding = 0), #144x96x64
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2), #72x48x64
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 1, stride=1, padding = 0), #72x48x128
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size= 1, stride=1, padding = 0), #72x48x128
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2), #36x24x128
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 1, stride=1, padding = 0), #36x24x256
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size= 1, stride=1, padding = 0), #36x24x256
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(4, stride=4), #9x6x256

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 1, stride=1, padding = 0), #9x6x512
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size= 1, stride=1, padding = 0), #9x6x512
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=3) #3x2x512
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(3*2*512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )

    def forward_once(self, x):
        r, affine_grid = self.stnmod(x)
        output = self.cnn1(r)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    
# Load the training dataset
train_dataloader = DataLoader(siamese_dataset_train,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)
val_dataloader = DataLoader(siamese_dataset_valid,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    print('Using CPU')
else:
    print('Using GPU')


net = CFEN().to(device)
criterion = ContrastiveLoss()
CEloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)


alpha = 0.5
model = net
device = 'cuda:0'

history = []
epochs = 3
for epoch in range(epochs):
    
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
    
    model.train()
    
    train_loss = 0.0
    train_acc  = 0.0
    train_f1   = 0.0
    train_ra   = 0.0
    valid_loss = 0.0
    valid_acc  = 0.0
    valid_f1   = 0.0
    valid_ra   = 0.0

    for i, (img0, img1, label, class0, class1) in enumerate(tqdm(train_dataloader)):   

        img0, img1, label, class0, class1 = img0.to(device), img1.to(device), label.to(device), class0.to(device), class1.to(device)
        
        output0, output1 = net(img0, img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output0, output1, label)
        loss0 = CEloss(output0, class0)
        loss1 = CEloss(output1, class1)
        full_loss = alpha*loss_contrastive  + loss0  + loss1
        full_loss.backward()
        optimizer.step()
 
        inpt_sz = (img0.size(0) + img1.size(0)) / 2

        preds0 = output0.argmax(-1)
        correct0 = (preds0 == class0).sum().item()
        
        preds1 = output1.argmax(-1)
        correct1 = (preds1 == class1).sum().item()

        train_loss += full_loss.item() * inpt_sz
        
        correct_counts = correct0 + correct1
        
        acc = ((correct0/BATCH_SIZE) + (correct1/BATCH_SIZE)) / 2
        train_acc += acc * inpt_sz

        f1_0  = f1_score(class0.data.view_as(preds0).cpu(), preds0.cpu())
        f1_1 = f1_score(class1.data.view_as(preds1).cpu(), preds1.cpu())
        f1 = (f1_0 + f1_1) / 2
        train_f1  += f1 * inpt_sz
        try:
            ra_0  = roc_auc_score(class0.data.view_as(preds0).cpu(), preds0.cpu())
            ra_1 = roc_auc_score(class1.data.view_as(preds1).cpu(), preds1.cpu())
            ra = (ra_0 + ra_1) / 2
            train_ra  += ra *inpt_sz
        except ValueError:
            pass     
        # Train logs
        if i % 100 == 0:
            print(f"Batch number: {i:03d}, Training: Loss: {train_loss/(i+1)/inpt_sz:.4f}, Accuracy: {train_acc/(i+1)/inpt_sz:.4f}, F1: {train_f1/(i+1)/inpt_sz:.4f}, ROC-AUC: {train_ra/(i+1)/inpt_sz:.4f}")
    
    # Validation - No gradient tracking needed
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        # Validation loop
        for j, (img0, img1, label, class0, class1) in enumerate(val_dataloader):               
            img0, img1, label, class0, class1 = img0.to(device), img1.to(device), label.to(device), class0.to(device), class1.to(device)

            # Forward pass - compute outputs on input data using the model
            output0, output1 = net(img0, img1)
            
            inpt_sz = (img0.size(0) + img1.size(0)) / 2
            
            loss_contrastive = criterion(output0, output1, label)
            loss0 = CEloss(output0, class0)
            loss1 = CEloss(output1, class1)
            full_loss = alpha*loss_contrastive  + loss0  + loss1
            valid_loss += full_loss.item() * inpt_sz 
            
            preds0 = output0.argmax(-1)
            correct0 = (preds0 == class0).sum().item()

            preds1 = output1.argmax(-1)
            correct1 = (preds1 == class1).sum().item()
            
            correct_counts = correct0 + correct1

            acc = ((correct0/BATCH_SIZE) + (correct1/BATCH_SIZE)) / 2
            valid_acc += acc * inpt_sz

            f1_0  = f1_score(class0.data.view_as(preds0).cpu(), preds0.cpu())
            f1_1 = f1_score(class1.data.view_as(preds1).cpu(), preds1.cpu())
            f1 = (f1_0 + f1_1) / 2
            valid_f1  += f1 * inpt_sz
            try:
                ra_0  = roc_auc_score(class0.data.view_as(preds0).cpu(), preds0.cpu())
                ra_1 = roc_auc_score(class1.data.view_as(preds1).cpu(), preds1.cpu())
                ra = (ra_0 + ra_1) / 2
                valid_ra  += ra *inpt_sz
            except ValueError:
                pass
            
            # Valid logs
            if j % 100 == 0:
                print(f"Validation Batch number: {i:03d}, Validation: Loss: {valid_loss/(j+1)/inpt_sz:.4f}, Accuracy: {valid_acc/(j+1)/inpt_sz:.4f}, F1: {valid_f1/(j+1)/inpt_sz:.4f}, ROC-AUC: {valid_ra/(j+1)/inpt_sz:.4f}")

    avg_train_loss = train_loss/train_data_size 
    avg_train_acc  = train_acc/float(train_data_size)
    avg_train_f1   = train_f1/float(train_data_size)
    avg_train_ra   = train_ra/float(train_data_size)

    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc  = valid_acc/float(valid_data_size)
    avg_valid_f1   = valid_f1/float(valid_data_size)
    avg_valid_ra   = valid_ra/float(valid_data_size)

    name = f"CFEN_{epoch:03d}"
    torch.save(model, f"Checkpoints/{name}.pth")
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    print(f"Epoch : {epoch:03d}, Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%, F1: {avg_train_f1:.4f}, ROC-AUC: {avg_train_ra:.4f}\n Validation : Loss : {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.4f}%, F1: {avg_valid_f1:.4f}, ROC-AUC: {avg_valid_ra:.4f}, Time: {epoch_end-epoch_start:.4f}s")
