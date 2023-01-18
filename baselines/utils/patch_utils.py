import random
import torch
import torchvision
from torchvision import datasets, transforms
from torch.data.utils import DataLoader, Dataset
from PIL import Image
import numpy as np
import argparse
import time
import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from common_utils import get_device

device = get_device()

NUM_WORKERS = 4
SIZE_H = 144
SIZE_W = 96
BATCH_SIZE = 64
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


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



cfen_transformer = torchvision.transforms.Compose([
    transforms.Resize((SIZE_H, SIZE_W)),
    transforms.ToTensor(), 
    transforms.Normalize(image_mean, image_std)
])


def load_data(path: str, bs: int, image_transforms: dict):
    """Load data."""
    train_directory = f'{path}/patch_splitted/train'
    valid_directory = f'{path}/patch_splitted/val'
    test_directory = f'/{path}/patch_splitted/test'

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
    return train_data, valid_data, test_data

def load_cfen_data(path: str):

    NUM_WORKERS = 4
    SIZE_H = 144
    SIZE_W = 96
    BATCH_SIZE = 64
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

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


    train_dataset = torchvision.datasets.ImageFolder(f'{path}/patch_splitted/train')
    valid_dataset = torchvision.datasets.ImageFolder(f'{path}/patch_splitted/val')
    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)

    transformer = torchvision.transforms.Compose([
        transforms.Resize((SIZE_H, SIZE_W)),
        transforms.ToTensor(), 
        transforms.Normalize(image_mean, image_std)
    ])

    siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=train_dataset, transform=transformer)
    siamese_dataset_valid = SiameseNetworkDataset(imageFolderDataset=valid_dataset, transform=transformer)

    train_dataloader = DataLoader(siamese_dataset_train,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(siamese_dataset_valid,
                            num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE)

    return train_dataloader, val_dataloader


def parse_args():

    parser = argparse.ArgumentParser(description='Baseline training parameters')

    parser.add_argument('-p', '--path', type=str, help='path to the dataset')

    parser.add_argument('-m_t', '--model_type', type=str, help='model type')

    args = parser.parse_args()

    return args


def train_patch_model(model, loss_func, optimizer, train_data, valid_data):

    history = []
    epochs = 10

    train_data_size = len(train_data.dataset)
    valid_data_size = len(valid_data.dataset)

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

            # Train logs
            if i % 100 == 0:
                print(f"Batch number: {i:03d}, Training: Loss: {train_loss/(i+1)/inputs.size(0):.4f}, Accuracy: {train_acc/(i+1)/inputs.size(0):.4f}, F1: {train_f1/(i+1)/inputs.size(0):.4f}, ROC-AUC: {train_ra/(i+1)/inputs.size(0):.4f}")

        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()
            # Validation loop
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

                # Valid logs
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


        name = f"{epoch:03d}"
        torch.save(model, f"Checkpoints/{name}.pth")
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        print(f"Epoch : {epoch:03d}, Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%, F1: {avg_train_f1:.4f}, ROC-AUC: {avg_train_ra:.4f}\n Validation : Loss : {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.4f}%, F1: {avg_valid_f1:.4f}, ROC-AUC: {avg_valid_ra:.4f}, Time: {epoch_end-epoch_start:.4f}s")


def train_cfen_model(model, CEloss, criterion, optimizer, train_data, valid_data):

    alpha = 0.5

    history = []
    epochs = 3

    train_data_size = len(train_data.dataset)
    valid_data_size = len(valid_data.dataset)

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

        for i, (img0, img1, label, class0, class1) in enumerate(tqdm(train_data)):

            img0, img1, label, class0, class1 = img0.to(device), img1.to(device), label.to(device), class0.to(device), class1.to(device)
            
            output0, output1 = model(img0, img1)
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
            for j, (img0, img1, label, class0, class1) in enumerate(valid_data):
                img0, img1, label, class0, class1 = img0.to(device), img1.to(device), label.to(device), class0.to(device), class1.to(device)

                # Forward pass - compute outputs on input data using the model
                output0, output1 = model(img0, img1)
                
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

        name = f"{epoch:03d}"
        torch.save(model, f"Checkpoints/{name}.pth")
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        print(f"Epoch : {epoch:03d}, Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%, F1: {avg_train_f1:.4f}, ROC-AUC: {avg_train_ra:.4f}\n Validation : Loss : {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.4f}%, F1: {avg_valid_f1:.4f}, ROC-AUC: {avg_valid_ra:.4f}, Time: {epoch_end-epoch_start:.4f}s")