import numpy as np
import torch.nn as nn
from torchvision import models

class ResModel(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, nclasses)
        self.history = {'train_loss': [], 'train_accuracy': [],
                        'val_loss': [], 'val_accuracy': []}
        
    def forward(self, x):
        return self.network(x)