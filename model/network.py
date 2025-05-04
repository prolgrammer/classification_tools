import torch
import torch.nn as nn
import torchvision.models as models

class ToolsClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ToolsClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)