import torch
import wandb
from tqdm import tqdm
import os
import torch
import hydra

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader

cheeses = ["BEAUFORT", "BRIE DE MEULIN"]

def train(cfg):
    logger = wandb.init(project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    datamodule.dataset = ImageFolder(datamodule.train_dataset_path, transform=datamodule.train_transform)

    clstokeep = [datamodule.dataset.class_to_idx[cheese] for cheese in cheeses]

    idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in clstokeep]
    dataset = Subset(dataset, idx)
    datamodule.train_dataset, datamodule.val_dataset = torch.utils.data.random_split(
        datamodule.dataset,
        [
            int(0.8 * len(datamodule.dataset)),
            len(datamodule.dataset) - int(0.8 * len(datamodule.dataset)),
        ],
        generator=torch.Generator().manual_seed(3407),
    )
    datamodule.val_dataset.transform = datamodule.val_transform
    datamodule.real_images_val_dataset = ImageFolder(
        datamodule.real_images_val_path, transform=datamodule.val_transform
    )
    
    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in tqdm(enumerate(train_loader), leave=False):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        val_metrics = {}
        for val_set_name, val_loader in val_loaders.items():
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            y_true = []
            y_pred = []
            for i, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.argmax(1).detach().cpu().tolist())
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            val_metrics[f"{val_set_name}/loss"] = epoch_loss
            val_metrics[f"{val_set_name}/acc"] = epoch_acc
            val_metrics[f"{val_set_name}/confusion_matrix"] = (
                wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[
                        datamodule.idx_to_class[i][:10].lower()
                        for i in range(len(datamodule.idx_to_class))
                    ],
                )
            )

        logger.log(
            {
                "epoch": epoch,
                **val_metrics,
            }
        )
        os.makedirs(cfg.checkpoint_path[:-3], exist_ok = True) 
        torch.save(model.state_dict(), cfg.checkpoint_path[:-3] + f"/epoch_{epoch}.pt")
    torch.save(model.state_dict(), cfg.checkpoint_path)
    logger.finish()