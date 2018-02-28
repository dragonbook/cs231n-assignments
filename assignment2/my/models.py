import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*13*13, 1024)
        self.fc2 = nn.Linear(1024, 10)


    def flatten(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


    def forward(self, x):
        x = self.max_pool(self.bn(self.relu(self.conv(x))))
        x = self.flatten(x)
        x = self.fc2(self.fc1(x))
        return x
