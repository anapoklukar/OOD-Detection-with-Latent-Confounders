import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18WithFeatures(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits


def get_resnet18(num_classes=10):
    return ResNet18WithFeatures(num_classes=num_classes)