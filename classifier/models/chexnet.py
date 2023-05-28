import torch.nn as nn
from torchvision.models import densenet121


class ChexNet(nn.Module):
    def __init__(self, trained=False, model_name="20210223-095555"):
        super().__init__()

        self.backbone = densenet121(False).features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 14)
        )

    def forward(self, x):
        x = nn.Linear(3, 1)(x)
        return self.head(self.backbone(x))
