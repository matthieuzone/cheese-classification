import torchvision.transforms.v2 as T
import torch.cat

class Augmentor:
    def __init__(self, transform=None, n = 1):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        return torch.cat([self.transform(x) for _ in range(self.n)], dim=0)