import cv2
import easyocr
import matplotlib.pyplot as plt
import difflib
import os
import glob

# Lire la liste des fromages depuis un fichier
with open('list_of_cheese.txt', 'r', encoding='utf-8') as fichier:
    list_of_cheese = [ligne.strip().lower() for ligne in fichier]

# Initialiser le lecteur EasyOCR
reader = easyocr.Reader(['fr'])

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
    closest_matches = []
    for cheese in list_of_cheese:
        match, score = find_closest_match(cheese, lecture_image(img))
        if match:
            closest_matches.append((cheese, match, score))
    best_match = max(closest_matches, key=lambda x: x[2])
    return best_match
"""
closest_cheeses = find_closest_cheese(list_of_cheese)
print("closest:", closest_cheeses)
#for cheese, match, score in closest_cheeses:
    #print(f"Fromage: {cheese}, Correspondance trouvée: {match}, Score: {score:.2f}")

# Trouver et afficher le meilleur score
if closest_cheeses:
    best_match = max(closest_cheeses, key=lambda x: x[2])
    print(best_match)
    print(f"Meilleure correspondance: Fromage: {best_match[0]}, Correspondance trouvée: {best_match[1]}, Score: {best_match[2]:.2f}")
else:
    print("Aucune correspondance trouvée.")
"""
# Chemin du dossier principal
dossier_principal = '/Users/user/cheese-classification/dataset/val_with_text'

fichiers_images = glob.glob(os.path.join(dossier_principal, '**', '*.jpg'), recursive=True)
    
for fichier in fichiers_images:
    print(fichier, find_closest_cheese(list_of_cheese, cv2.imread(fichier)))