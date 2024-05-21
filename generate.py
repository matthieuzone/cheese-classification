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
    for label in labels:
        try:
            dataset_generator.generate(labels)
        except Exception as e:
            print(f"Error generating images for {label}: {e}")
            wandb.log({"error": f"Error generating images for {label}: {e}"})


if __name__ == "__main__":
    while psutil.pid_exists(1127025):
        time.sleep(60)
    generate()
