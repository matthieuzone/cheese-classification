import random
import os

repertoire = 'prompts/'

def choisir_adjectif(fichier):
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


def create_prompt(label) :
    syntaxe = "Créé une image de " + label.lower()+" "+ choisir_adjectif('adjectifs.txt').lower()+ " et " + choisir_adjectif('adjectifs.txt').lower() + " qui est vu " +choisir_adjectif('vue.txt')+ ", placé sur " +choisir_adjectif('arriere_plan.txt')
    return syntaxe

#print(syntaxe)