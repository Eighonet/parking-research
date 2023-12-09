import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



def get_cfen():

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

    

    net = CFEN()
    criterion = ContrastiveLoss()
    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    return net, criterion, CEloss, optimizer