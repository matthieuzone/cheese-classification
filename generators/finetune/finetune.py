import subprocess
import os
import time

with open("list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

def main():
    args = [
        '--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0',
        '--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix',
        '--mixed_precision=fp16',
        '--resolution=1024',
        '--train_batch_size=1',
        '--gradient_accumulation_steps=4',
        '--learning_rate=1e-4',
        '--report_to=wandb',
        '--lr_scheduler=constant',
        '--lr_warmup_steps=0',
        '--max_train_steps=20',
        #'--validation_epochs=25',
        '--enable_xformers_memory_efficient_attention',
        '--gradient_checkpointing',
        '--use_8bit_adam',
        '--push_to_hub'
        ]
    l = len(args)
    args.extend("" for _ in range(3))
    root_data_dir = "dataset/val"
    roor_output_dir = "checkpoints/finetune2/"
    labels = ["SAINT- FÉLICIEN", "COMTÉ"]
    for label in labels:
        if label not in ["BEAUFORT", "BRIE DE MEULIN", "CABECOU", "CAMEMBERT"]:
            args[l  ] = f'--instance_data_dir={root_data_dir}/{label}'
            args[l+1] = f'--instance_prompt=a photo of {label} cheese'
            #args[l+2] = f'--validation_prompt=A photo of {label} cheese on a plate'
            args[l+2] = f'--output_dir={roor_output_dir}/info2{label}'.replace(" ", "_").replace("Û", "U").replace("È", "E").replace("’", "_").replace("É", "E")
            subprocess.run(["accelerate", "launch", "generators/finetune/train_dreambooth_lora_sdxl.py"] + args)
    print("\n\nDone")
    
if __name__ == "__main__":  
    while not os.path.exists("checkpoints/finetune2/info2VACHERIN/README.md"):
        time.sleep(60)
    print("\n\nStarting FINE TUNING\n\n")
    main()


