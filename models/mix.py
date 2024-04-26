import dinov2
import resnet
import torch
import torch.nn as nn

class mixedModel(nn.Module):
    def __init__(self, p_dino, num_classes, frozen=False, unfreeze_last_layer=True):
        super(mixedModel, self).__init__()
        self.dino = dinov2.DinoV2Finetune(num_classes, frozen, unfreeze_last_layer)
        self.resnet = resnet.ResNet(num_classes, frozen)
        self.p_dino = p_dino

    def forward(self, x):
        return self.dino(x) * self.p_dino + self.resnet(x) * (1 - self.p_dino)