from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
from huggingface_hub import login
import os
import glob

token = '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'
login(token=token)

model_id = "google/paligemma-3b-mix-224"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open('maroilles.jpg')

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "caption with details"

def generate_caption(img, prompt):
    model_inputs = processor(text=prompt, images=img, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=150, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)  
    return decoded

# Chemin du dossier principal
dossier_principal = 'val'
"""
fichiers_images = glob.glob(os.path.join(dossier_principal, '**', '*.jpg'), recursive=True)
    
for fichier in fichiers_images:
    print(fichier, generate_caption(Image.open(fichier), prompt))
"""

# Traverse through all directories and subdirectories
for root, dirs, files in os.walk(dossier_principal):
    # Initialize a list to collect image paths and captions
    image_data = []
    print("new iteration")
    # Find all .jpg files in the current directory
    for file in files:
        if file.lower().endswith('.jpg'):
            file_path = os.path.join(root, file)
            try:
                # Open the image
                image = Image.open(file_path)
                # Generate the caption
                caption = generate_caption(image, prompt)
                # Collect the data
                image_data.append(f"{file_path}: {caption}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Write the collected data to a text file named after the current directory
    if image_data:
        relative_dir = os.path.relpath(root, dossier_principal)
        text_file_name = os.path.join(dossier_principal, f"{relative_dir.replace(os.sep, '_')}.txt")
        try:
            with open(text_file_name, 'w', encoding='utf-8') as text_file:
                for line in image_data:
                    text_file.write(line + "\n")
                print("new file done")
        except Exception as e:
            print(f"Error writing to file {text_file_name}: {e}")
