import torch
import torch.nn as nn
from aster.model import Aster  # Import the ASTER model class from the appropriate module

class ASTERFinetune(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, frozen=False, unfreeze_last_layer=True):
        super().__init__()

        # Load the pre-trained ASTER model
        self.model = Aster(num_classes=num_classes)  # Initialize the ASTER model

        if pretrained_path:
            # Load weights from the pre-trained checkpoint
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(checkpoint['state_dict'])

        # Optionally freeze layers
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.model.decoder[-1].parameters():  # Assuming the last layer of the ASTER model is the decoder
                    param.requires_grad = True

        # Update the number of output classes in the final linear layer
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
