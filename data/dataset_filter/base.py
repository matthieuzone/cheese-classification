from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms

class DatasetFilterBase:

    def __init__(self, batch_size=64, transform = [], input_dir="dataset/train", output_dir="dataset/train"):
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
    
    def filter(self, labels_names):
        dataset = ImageFolder(self.input_dir, transform=torchvision.transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        image_ids = [0 for label in labels_names]
        for x, y in tqdm(dataloader):
            r = self.critere(x).to('cpu')
            x = x[r]
            y = y[r]
            self.save_images(x, labels_names, y, image_ids)
    
    def save_images(self, images, labels_names, labels_nums, image_ids):
        output_path = Path(self.output_dir)
        for name in labels_names:
            (output_path / name).mkdir(parents = True, exist_ok = True)
        for i, image in enumerate(images):
            image = torchvision.transforms.ToPILImage()(image)
            image.save(output_path / labels_names[labels_nums[i]] / f"{str(image_ids[labels_nums[i]]).zfill(6)}.jpg")
            image_ids[labels_nums[i]] += 1

    def critere(self, x):
        return NotImplementedError