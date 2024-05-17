import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub.repocard import RepoCard
import torchvision.transforms.functional

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = []
with open("../../../list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

class SDXLFineTunedGenerator:
    def __init__(
        self,
    ):
        self.num_inference_steps = 30
        self.guidance_scale = 8

        #self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
        self.active_label = None

    def initpipe(self, label):
        print("\nLoading model for", label)
        lora_model_id = f'matthieuzone/{label}2'.replace(" ", "_").replace("Û", "U").replace("È", "E").replace("’", "_").replace("É", "E")
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]

        # Load the base pipeline and load the LoRA parameters into it. 
        self.pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        self.pipe.load_lora_weights(lora_model_id)

        # Load the refiner.
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.refiner.to("cuda")
        self.generator = torch.Generator("cuda")
        self.active_label = label

    def generate(self, prompts, label):
        if label != self.active_label:
            self.initpipe(label)
        print("\nsdxl Generating")
        images = self.pipe(
            prompts,
            output_type="latent",
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
        ).images
        print("\nRefining")
        images = self.refiner(prompt=prompts, image = images).images
        return images