import torch
import wandb
import hydra
from tqdm import tqdm
import psutil
import time


@hydra.main(config_path="configs/generate", config_name="config", version_base=None)
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    a = 35
    labels = [labels[7],labels[36],labels[25],labels[15]]

    for label in labels:
        print(f"Generating images for {label}")
        try:
            dataset_generator.generate(labels)
        except Exception as e:
            print(f"Error generating images for {label}: {e}")
            wandb.log({"error": f"Error generating images for {label}: {e}"})
        print(f"Generated images for {label}")


if __name__ == "__main__":
    generate()
