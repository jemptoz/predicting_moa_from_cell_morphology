import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
from pathlib import Path

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "resnet18-f37072fd.pth"

class MoAResNet(nn.Module):
    """ResNet-18 model for predicting mechanism of action from microscopy images."""

    def __init__(self, num_classes=12, weights_path=None):
        super().__init__()

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH

        self.model = resnet18(weights=None)

        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)


        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)