import hydra

@hydra.main(config_path="configs/filter", config_name="config")
def filter(cfg):
    fil = hydra.utils.instantiate(cfg.critere)
    
    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    fil.filter()

if __name__ == "__main__":
    filter()