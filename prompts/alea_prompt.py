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

syntaxe = "Créé une image de " + choisir_fromage('list_of_cheese.txt').lower()+" "+ choisir_adjectif('adjectifs.txt').lower()+ " et " + choisir_adjectif('adjectifs.txt').lower() + " qui est vu " +choisir_adjectif('vue.txt')+ ", placé sur " +choisir_adjectif('arriere_plan.txt')

syntaxe_eng_simple = "Create an image of the cheese '" + choisir_fromage('list_of_cheese.txt').lower()+"'"
syntaxe_eng_complex = "Create an image of the cheese '" + choisir_fromage('list_of_cheese.txt').lower()+"' "+ choisir_adjectif('adjectifs_eng.txt').lower()+ " and " + choisir_adjectif('adjectifs_eng.txt').lower() + " seen from " +choisir_adjectif('vue_eng.txt')+ ", placed on " +choisir_adjectif('arriere_plan_eng.txt')

def create_prompt(label,repertoire) :
    syntaxe = "Créé une image de " + label.lower()+" "+ choisir_adjectif('adjectifs.txt',repertoire).lower()+ " et " + choisir_adjectif('adjectifs.txt',repertoire).lower() + " qui est vu " +choisir_adjectif('vue.txt',repertoire)+ ", placé sur " +choisir_adjectif('arriere_plan.txt',repertoire)+ " et a une texture " + choisir_adjectif('texture.txt',repertoire).lower()+"."
    return syntaxe

#print(syntaxe)