import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLImg2ImgPipeline
from huggingface_hub.repocard import RepoCard

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = []
with open("list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

class SDXLLightiningGenerator:
    def __init__(
        self,
    ):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipes = {}
        for label in labels:
            self.pipes[label] = DiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16).to(device)
            self.pipes[label].scheduler = EulerDiscreteScheduler.from_config(
                self.pipes[label].scheduler.config, timestep_spacing="trailing"
            )
            self.pipes[label].load_lora_weights("checkpoints/finetune/"+label)
            self.pipes[label].set_progress_bar_config(disable=True)
        self.num_inference_steps = 4
        self.guidance_scale = 0

        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )
        self.refiner.to(device)

    def generate(self, prompts, label):
        images = self.pipes[label](
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        images = self.refiner(prompt=prompts, image = images)
        return images