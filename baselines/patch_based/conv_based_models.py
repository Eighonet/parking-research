import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet50, mobilenetv2
import geffnet

num_classes = 2

def get_alexnet():

    alexnet = torchvision.models.alexnet(pretrained=True)

    for param in alexnet.parameters():
        param.requires_grad = False

    fc_inputs = alexnet.classifier[1].in_features
    alexnet.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(alexnet.parameters())

    return alexnet, loss_func, optimizer

def get_carnet():

    class CarNet(nn.Module):
        def __init__(self):
            super(CarNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 96, 11, dilation=2, padding=(11,10))
            self.conv2 = nn.Conv2d(96, 192, 11, dilation=2, padding=(10, 10))
            self.conv3 = nn.Conv2d(192, 384, 11, dilation=2, padding=(12, 10))
            self.fc1 = nn.Linear(18432, 4096)  # 5*5 from image dimension
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, 2)
            self.dropout = nn.Dropout(p=0.4)
            self.output = nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = F.interpolate(x, size=52)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)  
            x = torch.flatten(x, 1)
            x = F.relu(self.dropout(self.fc1(x)))
            x = F.relu(self.dropout(self.fc2(x)))
            x = self.output(self.fc3(self.dropout(x)))
            return x

    model = CarNet()

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)

    return model, loss_func, optimizer

def get_efficientnet():

    model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)

    for param in model.parameters():
        param.requires_grad = True

    model.classifier = nn.Linear(in_features=1280, out_features=2, bias=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    return model, loss_func, optimizer

def get_malexnet():

    class mAlexNet(nn.Module):
        def __init__(self, num_classes = 2):
            super(mAlexNet, self).__init__()
            self.input_channel = 3
            self.num_output = num_classes
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channel, out_channels= 16, kernel_size= 11, stride= 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels= 16, out_channels= 20, kernel_size= 5, stride= 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels= 20, out_channels= 30, kernel_size= 3, stride= 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(30*3*3, out_features=48),
                nn.ReLU(inplace=True)
            )
            self.fc2 = nn.Linear(in_features=48, out_features=2)

        def forward(self, x):
            x = self.conv3(self.conv2(self.conv1(x)))
            x = x.view(x.size(0), -1)
            x = self.fc2(self.fc1(x))
            return F.log_softmax(x, dim = 1)

    malexnet = mAlexNet()

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(malexnet.parameters())

    return malexnet, loss_func, optimizer

def get_mobilenet():

    # !git clone https://github.com/d-li14/mobilenetv2.pytorch.git
    model = mobilenetv2()
    model.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))

    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Linear(in_features=1280, out_features=2, bias=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    return model, loss_func, optimizer

def get_resnet50():

    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = True

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    return resnet50, loss_func, optimizer

def get_vgg16():

    vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)

    for param in vgg16_bn.parameters():
        param.requires_grad = False

    fc_inputs = vgg16_bn.classifier[0].in_features
    vgg16_bn.classifier = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(vgg16_bn.parameters())

    return vgg16_bn, loss_func, optimizer

def get_vgg19():

    vgg19 = torchvision.models.vgg19(pretrained=True)

    for param in vgg19.parameters():
        param.requires_grad = False

    fc_inputs = vgg19.classifier[0].in_features
    vgg19.classifier = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(vgg19.parameters())

    return vgg19, loss_func, optimizer