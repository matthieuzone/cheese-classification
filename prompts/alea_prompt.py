import random
import os

repertoire = 'prompts'

list_of_cheese = []
with open(os.path.join('../../../list_of_cheese.txt'), 'r') as f:
    list_of_cheese += f.readlines()

def choisir(fichier,repertoire,label):
    adjectifs = []
    lines = []
    with open(os.path.join(repertoire, fichier), 'r') as f:
        lines += f.readlines()
    for line in lines:
        line = line.split(",")
        line = [x.strip() for x in line]
        if label in line:
            for i in range(5):
                adjectifs.append(line[0])
        if "ALL" in line:
            adjectifs.append(line[0])
    return random.choice(adjectifs).strip()


#fichiers = [os.path.join(repertoire, 'adjectifs.txt'), os.path.join(repertoire, 'texture.txt'), os.path.join(repertoire, 'vue.txt'), os.path.join(repertoire, 'arriere_plan.txt')]
#syntaxe = "Créé une image de " + choisir_fromage('list_of_cheese.txt').lower()+" "+ choisir_adjectif('adjectifs.txt').lower()+ " et " + choisir_adjectif('adjectifs.txt').lower() + " qui est vu " +choisir_adjectif('vue.txt')+ ", placé sur " +choisir_adjectif('arriere_plan.txt')
#syntaxe_eng_simple = "Create an image of the cheese '" + choisir_fromage('list_of_cheese.txt').lower()+"'"
#syntaxe_eng_complex = "Create an image of the cheese '" + choisir_fromage('list_of_cheese.txt').lower()+"' "+ choisir_adjectif('adjectifs_eng.txt').lower()+ " and " + choisir_adjectif('adjectifs_eng.txt').lower() + " seen from " +choisir_adjectif('vue_eng.txt')+ ", placed on " +choisir_adjectif('arriere_plan_eng.txt')

def create_prompt(label, repertoire):
    a = "a picture of a"
    b = " " + choisir('before.txt',repertoire, label).lower()
    b2 = " " + choisir('before.txt',repertoire, label).lower()
    c = " " + label + " cheese"
    d = " " + choisir('after.txt',repertoire, label).lower()
    d2 = " " + choisir('after.txt',repertoire, label).lower()
    syntaxe = a
    p = 0.8
    p2 = 0.3
    if random.random() > 1-p:
        syntaxe += b
    if random.random() > 1-p2:
        syntaxe += b2
    syntaxe += c
    if random.random() > 1-p:
        syntaxe += d
    if random.random() > 1-p2:
        syntaxe += d2
    return syntaxe

#print(syntaxe)