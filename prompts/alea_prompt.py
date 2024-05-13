import random
import os

repertoire = 'prompts/'

def choisir_adjectif(fichier,repertoire):
    adjectifs = []
    with open(os.path.join(repertoire, fichier), 'r') as f:
        adjectifs += f.readlines()
    return random.choice(adjectifs).strip()

def choisir_fromage(fichier):
    fromages = []
    with open(fichier, 'r') as f:
        fromages += f.readlines()
    return random.choice(fromages).strip()

fichiers = [os.path.join(repertoire, 'adjectifs.txt'), os.path.join(repertoire, 'texture.txt'), os.path.join(repertoire, 'vue.txt'), os.path.join(repertoire, 'arriere_plan.txt')]


def create_prompt(label,repertoire) :
    syntaxe = "Créé une image de " + label.lower()+" "+ choisir_adjectif('adjectifs.txt',repertoire).lower()+ " et " + choisir_adjectif('adjectifs.txt',repertoire).lower() + " qui est vu " +choisir_adjectif('vue.txt',repertoire)+ ", placé sur " +choisir_adjectif('arriere_plan.txt',repertoire)+ " et a une texture " + choisir_adjectif('texture.txt',repertoire).lower()+"."
    return syntaxe

#print(syntaxe)