import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(32*13*13, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )
        

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class BaseNet1(nn.Module):
    def __init__(self):
        super(BaseNet1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(32*13*13, 1024),
            #nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )
        

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class BatchNormNet(nn.Module):
    def __init__(self):
        super(BatchNormNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(32*13*13, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
        

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)    


class BatchNormNet1(nn.Module):
    def __init__(self):
        super(BatchNormNet1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(32*13*13, 1024),
            nn.BatchNorm1d(1024),
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
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )


    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class DifferentFilterNumberNet(nn.Module):
    def __init__(self, num_filters=32):
        super(DifferentFilterNumberNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(2*num_filters*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )


    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
