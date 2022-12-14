import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
import torchvision.models as models
from torchsampler import ImbalancedDatasetSampler
from models.imagenet import mobilenetv2
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
import time
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Baseline training parameters')

    parser.add_argument('-p', '--path', type=str, help='path to the dataset')

    args = parser.parse_args()

    return args

args = parse_args()


image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


train_directory = f'{args.path}/patch_splitted/train'
valid_directory = f'{args.path}/patch_splitted/val'
test_directory = f'/{args.path}/patch_splitted/test'

bs = 32
num_classes = 2
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
print("Load train data: ", train_data_size)
train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
print("Load valid data: ", valid_data_size)
valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
print("Load test data: ", test_data_size)
test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)


# !git clone https://github.com/d-li14/mobilenetv2.pytorch.git
model = mobilenetv2()
model.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))

for param in model.parameters():
    param.requires_grad = True
model.classifier = nn.Linear(in_features=1280, out_features=2, bias=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    print('Using CPU')
else:
    print('Using GPU')

model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

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
    for i, (inputs, labels) in enumerate(tqdm(train_data)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)

        f1  = f1_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
        train_f1  += f1 * inputs.size(0)

        try:
            ra  = roc_auc_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
            train_ra  += ra * inputs.size(0)
        except ValueError:
            print("ROC-AUC ValueError")

        if i % 10 == 0:
            print(f"Batch number: {i:03d}, Training: Loss: {train_loss/(i+1)/inputs.size(0):.4f}, Accuracy: {train_acc/(i+1)/inputs.size(0):.4f}, F1: {train_f1/(i+1)/inputs.size(0):.4f}, ROC-AUC: {train_ra/(i+1)/inputs.size(0):.4f}")

    # Validation - No gradient tracking needed
    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)

            f1  = f1_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
            valid_f1  += f1 * inputs.size(0)

            try:
                ra  = roc_auc_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
                valid_ra  += ra * inputs.size(0)
            except ValueError:
                print("ROC-AUC ValueError")

            if j % 100 == 0:
                print(f"Validation Batch number: {i:03d}, Validation: Loss: {valid_loss/(j+1)/inputs.size(0):.4f}, Accuracy: {valid_acc/(j+1)/inputs.size(0):.4f}, F1: {valid_f1/(j+1)/inputs.size(0):.4f}, ROC-AUC: {valid_ra/(j+1)/inputs.size(0):.4f}")

    avg_train_loss = train_loss/train_data_size 
    avg_train_acc  = train_acc/float(train_data_size)
    avg_train_f1   = train_f1/float(train_data_size)
    avg_train_ra   = train_ra/float(train_data_size)

    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc  = valid_acc/float(valid_data_size)
    avg_valid_f1   = valid_f1/float(valid_data_size)
    avg_valid_ra   = valid_ra/float(valid_data_size)
    
    name = f"MobileNet_{epoch:03d}"
    torch.save(model, f"Checkpoints/{name}.pth")
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    print(f"Epoch : {epoch:03d}, Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%, F1: {avg_train_f1:.4f}, ROC-AUC: {avg_train_ra:.4f}\n Validation : Loss : {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.4f}%, F1: {avg_valid_f1:.4f}, ROC-AUC: {avg_valid_ra:.4f}, Time: {epoch_end-epoch_start:.4f}s")