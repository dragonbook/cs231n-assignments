import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg


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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class ExampleDropoutNet(nn.Module):
    def __init__(self):
        super(ExampleDropoutNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(32*13*13, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )


    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class SmallFilterNet(nn.Module):
    def __init__(self):
        super(SmallFilterNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(64*8*8, 1024),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class SmallFilterDropoutNet(nn.Module):
    def __init__(self):
        super(SmallFilterDropoutNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(64*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


