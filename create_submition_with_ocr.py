import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
import cv2
import easyocr
import difflib

# Lire la liste des fromages depuis un fichier
with open('list_of_cheese.txt', 'r', encoding='utf-8') as fichier:
    list_of_cheese = [ligne.strip().lower() for ligne in fichier]

# Initialiser le lecteur EasyOCR
reader = easyocr.Reader(['fr'])

# Lire l'image

# Effectuer l'OCR sur l'image

def lecture_image(img):#lis l'image et renvoie la liste de mots lus
    liste_mots_image = []
    results = reader.readtext(img)
    for res in results:
        det, conf = res[1], res[2]
        liste_mots_image.append(det.lower())  # Convertir en minuscules pour une comparaison plus facile
    return liste_mots_image

def find_closest_match(word, word_list):
    closest_match = difflib.get_close_matches(word, word_list, n=1, cutoff=0)
    if closest_match:
        match = closest_match[0]
        score = difflib.SequenceMatcher(None, word, match).ratio()
        return match, score
    else:
        return None, 0.0
    

def find_closest_cheese(list_of_cheese, img):
    liste_mots = lecture_image(img)
    closest_matches = []
    for cheese in list_of_cheese:
        match, score = find_closest_match(cheese, liste_mots)
        if match:
            closest_matches.append((cheese, match, score))
    return closest_matches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    val_loader = datamodule.val_dataloader()['real_val']
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    def predict_image(images, image_names=None):
        preds = model(images)
        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        images = images.permute(0, 2, 3, 1)
        images = images.cpu().numpy()
        for i in range(images.shape[0]):
            matchs = find_closest_cheese(list_of_cheese, cv2.imread("../../../dataset/test/" + image_names[i]+".jpg"))
            if matchs:
                fromage, r, score = max(matchs, key=lambda x: x[2])
                if score > 0.6:
                    preds[i] = fromage.upper()
        return preds
    
    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = predict_image(images, image_names)
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission3.csv", index=False)
    """    
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        labels = [class_names[label] for label in labels.cpu().numpy()]
        preds = predict_image(images, )
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                correct += 1
            else:
                print(f"Predicted: {preds[i]}, True: {labels[i]}")

    print(f"Accuracy on real validation set: {correct / total}")
    """
if __name__ == "__main__":
    create_submission()
