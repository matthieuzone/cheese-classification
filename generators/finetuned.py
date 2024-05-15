import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub.repocard import RepoCard

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = []
with open("../../../list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

class SDXLFineTunedGenerator:
    def __init__(
        self,
    ):
        vae_path = "madebyollin/sdxl-vae-fp16-fix"
        self.num_inference_steps = 25
        self.guidance_scale = 0

        #self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
        self.active_label = None
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder=None,
            revision=None,
            variant=None,
            torch_dtype = torch.float16,
        )

    def initpipe(self, label):
        print("Loading model for", label)
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            vae=self.vae,
            revision=None,
            variant=None,
            torch_dtype=torch.float16,
        ).to(device)

        self.pipe.load_lora_weights("../../../checkpoints/finetune/"+label)
        self.pipe.set_progress_bar_config(disable=True)
        self.active_label = label

    def generate(self, prompts, label):
        if label != self.active_label:
            self.initpipe(label)
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        #images = self.refiner(prompt=prompts, image = images).images
        return images