import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.linear1 = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.linear2 = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        return x

    def forward(self, x1, x2):
        y1 = self.forward_one(x1)
        y2 = self.forward_one(x2)
        distance = torch.abs(y1 - y2)
        out = self.linear2(distance)
        #  return self.sigmoid(out)
        return out


class SiameseLeNet5(nn.Module):
    def __init__(self):
        super(SiameseLeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  
            nn.MaxPool2d(2),   
        )
        self.linear1 = nn.Sequential(nn.Linear(8464, 120), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.linear3 = nn.Linear(84, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def forward(self, x1, x2):
        y1 = self.forward_one(x1)
        y2 = self.forward_one(x2)
        distance = torch.abs(y1 - y2)
        out = self.linear3(distance)
        #  return self.sigmoid(out)
        return out  


