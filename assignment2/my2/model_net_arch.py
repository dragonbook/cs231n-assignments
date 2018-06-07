import torch
import torch.nn as nn


# like vgg network pattern
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A1': [32, 'M', 64, 'M'],
    'A2': [32, 'M', 64, 'M', 128, 'M'],
    'B1': [32, 32, 'M', 64, 64, 'M'],
    'B2': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
}

cfg_with_feature_size = {
    'A1': (cfg['A1'], (64, 8, 8)),
    'A2': (cfg['A2'], (128, 4, 4)),
    'B1': (cfg['B1'], (64, 8, 8)),
    'B2': (cfg['B2'], (128, 4,  4))
}


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class GlobalAveragePool2d(nn.Module):
    def __init__(self, h, w):
        super(GlobalAveragePool2d, self).__init__()
        self.gap = nn.AvgPool2d(kernel_size=(h, w), stride=(h, w))
        

    def forward(self, x):
        return self.gap(x)


class BaseClassiferHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(BaseClassiferHead, self).__init__()
        C, H, W = feature_size
        size = C * H * W

        self.classifier_head = nn.Sequential(
            Flatten(),
            nn.Linear(size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier_head(x)


class BaseDropoutClassiferHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(BaseDropoutClassiferHead, self).__init__()
        C, H, W = feature_size
        size = C * H * W

        self.classifier_head = nn.Sequential(
            Flatten(),
            nn.Linear(size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier_head(x)


class AlexNetStyleClassiferHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(AlexNetStyleClassiferHead, self).__init__()
        C, H, W = feature_size
        size = C * H * W
        
        self.classifier_head = nn.Sequential(
            Flatten(),
            nn.Dropout(),
            nn.Linear(size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        return self.classifier_head(x)


class VggNetStyleClassifierHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(VggNetStyleClassifierHead, self).__init__()
        C, H, W = feature_size
        size = C * H * W

        self.classifier_head = nn.Sequential(
            Flatten(),
            nn.Linear(size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        return self.classifier_head(x)


class SqueezeNetStyleClassifierHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(SqueezeNetStyleClassifierHead, self).__init__()
        C, H, W = feature_size
        size = C * H * W
        final_conv = nn.Conv2d(C, num_classes, kernel_size=1)

        self.classifier_head = nn.Sequential(
            nn.Dropout(),
            final_conv,
            nn.ReLU(inplace=True),
            GlobalAveragePool2d(H, W),
            Flatten()
        )


    def forward(self, x):
        return self.classifier_head(x)


class ResNetStyleClassifierHead(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(ResNetStyleClassifierHead, self).__init__()
        C, H, W = feature_size

        self.classifier_head = nn.Sequential(
            GlobalAveragePool2d(H, W),
            Flatten(),
            nn.Linear(C, num_classes)
        )


    def forward(self, x):
        return self.classifier_head(x)


class DepthNetBase(nn.Module):
    def __init__(self, features, classifier_head):
        super(DepthNetBase, self).__init__()
        self.features = features
        self.classifier = classifier_head

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_classifier_head(style, feature_size, num_classes):
    if style == 'base':
        return BaseClassiferHead(feature_size, num_classes)
    elif style == 'dropout':
        return BaseDropoutClassiferHead(feature_size, num_classes)
    elif style == 'alex':
        return AlexNetStyleClassiferHead(feature_size, num_classes)
    elif style == 'vgg':
        return VggNetStyleClassifierHead(feature_size, num_classes)
    elif style == 'squeeze':
        return SqueezeNetStyleClassifierHead(feature_size, num_classes)
    elif style == 'resnet':
        return ResNetStyleClassifierHead(feature_size, num_classes)
    else:
        return BaseClassiferHead(feature_size, num_classes)


def DepthNet(feature_body_style='A1', classifier_head_style='base', batch_norm=True):
    num_classes = 10
    features_cfg, feature_size = cfg_with_feature_size[feature_body_style]

    features = make_layers(features_cfg, batch_norm=batch_norm)
    classifier_head = make_classifier_head(classifier_head_style, feature_size, num_classes)

    model = DepthNetBase(features, classifier_head)

    return model
