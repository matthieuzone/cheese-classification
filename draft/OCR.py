import cv2
import easyocr
import matplotlib.pyplot as plt
import difflib

# Ouvrir le fichier en mode lecture
with open('list_of_cheese.txt', 'r', encoding='utf-8') as fichier:
    # Lire toutes les lignes du fichier et les stocker dans une liste
    list_of_cheese = [ligne.strip().lower() for ligne in fichier]

# Afficher la liste des noms
print(list_of_cheese)


# This needs to run only once to load the model into memory
reader = easyocr.Reader(['fr'])

# reading the image
img = cv2.imread('maroilles.jpg')

# run OCR
results = reader.readtext(img)
liste_mots = []

# show the image and plot the results
plt.imshow(img)
for res in results:
    # bbox coordinates of the detected text
    xy = res[0]
    xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]
    # text results and confidence of detection
    det, conf = res[1], res[2]
    liste_mots.append(det)
    # show time :)
    plt.plot([xy1[0], xy2[0], xy3[0], xy4[0], xy1[0]], [xy1[1], xy2[1], xy3[1], xy4[1], xy1[1]], 'r-')
    plt.text(xy1[0], xy1[1], f'{det} [{round(conf, 2)}]')

def find_closest_match(word, word_list):
    closest_match = difflib.get_close_matches(word, word_list, n=1)
    if closest_match:
        return closest_match[0]
    else:
        return None
    
def find_closest_cheese(list_of_cheese):
    closest = []
    for cheese in list_of_cheese:
        closest.append((cheese, find_closest_match(cheese, liste_mots)))
    return closest

print(find_closest_cheese(list_of_cheese))

