import generators.finetune.train_dreambooth_lora_sdxl as t
import argparse

with open("list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]


def main(args):
    args.pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"
    args.mixed_precision="fp16"
    args.resolution=1024
    args.train_batch_size=1
    args.gradient_accumulation_steps=4
    args.learning_rate=1e-4
    args.report_to="wandb"
    args.lr_scheduler="constant"
    args.lr_warmup_steps=0
    args.max_train_steps=500
    args.validation_epochs=25
    root_data_dir = "datasets/val"
    roor_output_dir = "checkpoints/finetune/"
    for label in labels:
        args.instance_data_dir = f"{root_data_dir}/{label}"
        args.instance_prompt = f"a photo of {label} cheese"
        args.validation_prompt = f"A photo of {label} on a plate"
        args.output_dir = f"{roor_output_dir}/{label}"
        t.main(args)
    
    
if __name__ == "__main__":
    args = t.parse_args()
    main(args)

