import torch
import wandb
import hydra
from tqdm import tqdm
import os
from omegaconf import OmegaConf

def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    dataset_generator.generate(labels)

def filter(cfg):
    fil = hydra.utils.instantiate(cfg.critere)
    
    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    fil.filter(labels)

def train(cfg):
    logger = wandb.init(project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
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

@hydra.main(config_path="configs/generate", config_name="config")
def gen(cfg):
    generate(cfg)
    cfg.dataset_generator.output_dir = "../../../datasets/random_prompts"
    cfg.augmentor = None
    generate(cfg)

@hydra.main(config_path="configs/filter", config_name="config")
def fil(cfg):
    filter(cfg)

def continue_training(cfg, id, epoch_init = 0, epoch_end = 20):
    logger = wandb.init(project="challenge_cheese", id=id, resume="must")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path))

    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(epoch_init, epoch_end)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
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

@hydra.main(config_path="configs/train", config_name="config")
def tr(cfg):
    cfg.dataset_name = "random_prompts_finetuned"
    cfg.model = OmegaConf.load("../../../configs/train/model/dinov2.yaml")
    cfg.epochs = 10

    try:
        cfg.experiment_name = "dinov2_0layer_unfrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_0layer_unfrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 0
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_0layer_unfrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_1layer_unfrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_1layer_unfrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 1
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_1layer_unfrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_2layer_unfrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_2layer_unfrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 2
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_2layer_unfrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_0layer_semifrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_0layer_semifrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = True
        cfg.model.instance.layers = 0
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_0layer_semifrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_1layer_semifrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_1layer_semifrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = True
        cfg.model.instance.layers = 1
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_1layer_semifrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_2layer_semifrozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_2layer_semifrozen.pt"
        cfg.model.instance.frozen = False
        cfg.model.instance.unfreeze_last_layer = True
        cfg.model.instance.layers = 2
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_2layer_semifrozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_0layer_frozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_0layer_frozen.pt"
        cfg.model.instance.frozen = True
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 0
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_0layer_frozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_1layer_frozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_1layer_frozen.pt"
        cfg.model.instance.frozen = True
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 1
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_1layer_frozen failed: {str(e)}")

    try:
        cfg.experiment_name = "dinov2_2layer_frozen"
        cfg.checkpoint_path = "../../../checkpoints/dinov2_2layer_frozen.pt"
        cfg.model.instance.frozen = True
        cfg.model.instance.unfreeze_last_layer = False
        cfg.model.instance.layers = 2
        train(cfg)
    except Exception as e:
        print(f"Experiment dinov2_2layer_frozen failed: {str(e)}")




if __name__ == "__main__":
    tr()