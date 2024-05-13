from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

# Liste de noms de fromages (à titre d'exemple)
fromages = ["Cheddar", "Gouda", "Brie", "Camembert", "Roquefort", "Feta", "Mozzarella"]

# Taille de l'image
largeur = 300
hauteur = 200

# Couleur de l'emballage (à titre d'exemple, vous pouvez choisir une couleur ou un motif)
couleur_emballage = (255, 255, 200)

# Charger une police de texte
police = ImageFont.load_default()

# Créer une image
image = Image.new("RGB", (largeur, hauteur), couleur_emballage)
dessin = ImageDraw.Draw(image)

# Choisir un nom de fromage aléatoire
nom_fromage = random.choice(fromages)

# Obtenir la taille du texte
texte_bbox = dessin.textbbox((0, 0), nom_fromage, font=police)
texte_largeur = texte_bbox[2] - texte_bbox[0]
texte_hauteur = texte_bbox[3] - texte_bbox[1]

# Ajouter le nom du fromage sur l'emballage
position_texte = ((largeur - texte_largeur) // 2, (hauteur - texte_hauteur) // 2)
dessin.text(position_texte, nom_fromage, fill=(0, 0, 0), font=police)

# Enregistrer l'image
image_path = "fromage_emballé.jpg"
image.save(image_path)

# Afficher l'image en utilisant Matplotlib
image_affichage = Image.open(image_path)
plt.imshow(image_affichage)
plt.axis('off')  # Masquer les axes
plt.show()