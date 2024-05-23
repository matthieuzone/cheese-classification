import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True, layers = 0):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
        n = self.backbone.norm.normalized_shape[0]
        if layers == 0:
            self.classifier = nn.Linear(n, num_classes)
        else:
            hidlayers = (nn.Sequential(nn.Linear(n, n), nn.ReLU()) for _ in range(1))
            self.classifier = nn.Sequential(
                *hidlayers,
                nn.Linear(n, num_classes)
            )
        print(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
