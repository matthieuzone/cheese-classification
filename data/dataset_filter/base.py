from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision

class DatasetFilterBase:

    def __init__(self, batch_size=64, transform = [], input_dir="dataset/train", output_dir="dataset/train"):
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
    
    def filter(self, labels_names):
        dataset = ImageFolder(self.input_dir, transform=torchvision.ToTensor())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for label in labels_names:
            image_id_0 = 0
            for x, _ in tqdm(dataloader):
                x = x[self.critere(x)]
                self.save_images(x, label, image_id_0)
                image_id_0 += len(x)
    
    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")

    def critere(self, x):
        return NotImplementedError