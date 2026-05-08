from __future__ import annotations

import torch


class SimpleClassifier(torch.nn.Module):
    def __init__(self, in_features: int = 12, num_classes: int = 3) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
