import torch
import torch.nn as nn
import torchvision


class VITFinetune(nn.Module):
    def __init__(self, num_classes, size = 'B', patch_size = 16, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        if size == 'B':
            if patch_size == 16:
                weights = torchvision.models.ViT_B_16_Weights.DEFAULT
                self.model = torchvision.models.vit_b_16(weights = weights)
            elif patch_size == 32:
                weights = torchvision.models.ViT_B_32_Weights.DEFAULT
                self.model = torchvision.models.vit_b_32(weights = weights)
            else:
                raise ValueError('Invalid patch size')
        elif size == 'L':
            if patch_size == 16:
                weights = torchvision.models.ViT_L_16_Weights.DEFAULT
                self.model = torchvision.models.vit_l_16(weights = weights)
            elif patch_size == 32:
                weights = torchvision.models.ViT_L_32_Weights.DEFAULT
                self.model = torchvision.models.vit_l_32(weights = weights)
            else:
                raise ValueError('Invalid patch size')
        elif size == 'H':
            if patch_size == 14:
                weights = torchvision.models.ViT_H_14_Weights.DEFAULT
                self.model = torchvision.models.vit_h_14(weights = weights)
            else:
                raise ValueError('Invalid patch size')
        else:
            raise ValueError('Invalid model size')

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.model.encoder.layers[-1].parameters():
                    param.requires_grad = True
        self.model.heads = nn.Linear(self.model.heads[0].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
