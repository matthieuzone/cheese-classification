import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterPlusXL
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = []

base_model_path = "SG161222/RealVisXL_V1.0"
image_encoder_path = "../../../checkpoints/models/image_encoder"
ip_ckpt = "../../../checkpoints/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

with open("../../../list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

class IpGenerator:
    def __init__(
        self,
    ):
        self.num_inference_steps = 30

        #self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            )
        self.pipe = self.pipe.to(device)
        self.ip_model = IPAdapterPlusXL(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    def generate(self, prompts, label):
        inimgs = [Image.open(f"../../../dataset/val/{label}/{i:0>6d}.jpg").resize((512,512)) for i in range(25)]
        imgs = [inimgs[random.randint(0,24)] for i in range(len(prompts))]
        images = self.ip_model.generate(
            pil_image=imgs,
            prompt = prompts,
            num_inference_steps=self.num_inference_steps,
            num_samples=1,
            scale = 0.3
        )
        return images