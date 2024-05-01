from .base import DatasetFilterBase
import torchvision
import torch.nn as nn
import torch
from itertools import cycle
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierFilter(DatasetFilterBase):

    def __init__(self, model, batch_size=64, transform=[], input_dir="dataset/train", output_dir="dataset/train", true_dir="dataset/val", treshold = 0.5, from_checkpoint = False, num_epochs = 10, model_path="checkpoints/resnet_classifier.pt"):
        super().__init__(batch_size, transform, input_dir, output_dir)
        self.model = model
        self.model.to(device)
        self.treshold = treshold
        self.true_dir = true_dir
        if from_checkpoint:
            self.model.load_state_dict(torch.load(model_path))

        else:
            self.train(num_epochs)
            torch.save(self.model.state_dict(), model_path)
        self.model.eval()

    def critere(self, x):
        with torch.no_grad():
            x = self.transform(x)
            x = x.to(device)
            y = self.model(x)
            y = torch.sigmoid(y)
            return (y[:, 1] > self.treshold).squeeze()

    def train(self, num_epochs=10):

        self.model.train()
        transform = Compose([ToTensor(), self.transform])
        gen_data = torchvision.datasets.ImageFolder(self.input_dir, transform=transform)
        gen_dataloader = torch.utils.data.DataLoader(gen_data, batch_size=self.batch_size, shuffle=True)
        true_data = torchvision.datasets.ImageFolder(self.true_dir, transform=transform)
        true_dataloader = torch.utils.data.DataLoader(true_data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        for _ in tqdm(range(num_epochs)):
            for (xf, _), (xt, _) in zip	(gen_dataloader, cycle(true_dataloader)):
                xf, xt = xf.to(device), xt.to(device)
                optimizer.zero_grad()
                y_predf = self.model(xf)
                loss = criterion(y_predf, torch.zeros(y_predf.shape[0], 1, device=device))
                y_predt = self.model(xt)
                loss += criterion(y_predt, torch.ones(y_predt.shape[0], 1, device=device))
                loss.backward()
                optimizer.step()