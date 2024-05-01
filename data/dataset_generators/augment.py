import torchvision.transforms.v2 as T
import itertools

class Augmentor:
    def __init__(self, transform=None, n = 1):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        return list(itertools.chain.from_iterable([self.transform(x) for _ in range(self.n)]))