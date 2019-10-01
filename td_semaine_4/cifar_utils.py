from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import zipfile
import imageio

def charger_cifar(repertoire, etiquettes=None, max_par_etiquettes=None):
    """Charger l'ensemble de données CIFAR

    Paramètres
    ----------
    repertoire: Le répertoire où se trouvent les données
    etiquettes: Une liste contenant des nombres entiers de 0 à 9, précisant les classes à charger
                Par défaut etiquettes=None, ce qui est équivalent à etiquettes=[0,1,2,3,4,5,6,7,8,9]
    max_par_etiquettes: Maximum de données à charger par classe. Par défaut, max_par_etiquettes=None
                        et toutes les données sont chargées.
                        
    Retour
    ------
    Un couple x, y:
        x est une matrice numpy, dont chaque ligne correspond à une image concaténée en un vecteur 
        de 3*32*32=3072 dimensions
            - Les dimensions 0 à 1023 contiennent les valeurs du canal rouge
            - Les dimensions 1024 à 2047 contiennent les valeurs du canal vert
            - Les dimensions 2048 à 3071 contiennent les valeurs du canal bleu
        y est un vecteur contenant la classe de chaque image, soit un entier de 0 à len(etiquettes)-1    
    """
    if etiquettes is None:
         etiquettes = np.arange(10)
    images_list = [None] * len(etiquettes)
    labels_list = [None] * len(etiquettes)
    for i, val in enumerate(etiquettes):
        nom_fichier = repertoire + f'label{val}.zip'
        fichier_zip = zipfile.ZipFile(nom_fichier, "r")
        liste_png = [a for a in fichier_zip.namelist() if a[-4:]=='.png']
        if max_par_etiquettes is not None and len(liste_png) > max_par_etiquettes:
            liste_png = liste_png[:max_par_etiquettes]
         
        nb = len(liste_png)
        data = np.zeros((nb, 32*32*3), dtype=np.float32)
        for j, nom_image in enumerate(liste_png):
            buffer = fichier_zip.read(nom_image)
            image = imageio.imread(buffer)
            r, g, b = [image[:, :, c].reshape(-1) for c in (0,1,2)]
            data[j] = np.concatenate((r,g,b))
        
        images_list[i] = data / 255
        labels_list[i] = i*np.ones(nb, dtype=np.int64)
        print(val, ':', nb, 'images')
        
    x = np.vstack(images_list)
    y = np.concatenate(labels_list)
    print('Total :', len(y), 'images')
    return x, y


def afficher_grille_cifar(images):
    """Affiche une grille d'images provenant de l'ensemble CIFAR (en format numpy.array ou torch.Tensor). 
    Chaque image doit contenir 3*32*32=3072 pixels."""
    plt.figure(figsize=(15,6))
    images3d = torch.Tensor(images).view(-1,3,32,32)
    grid = make_grid(images3d)
    grid = grid.detach().numpy().transpose((1,2,0))
    plt.imshow(grid) 
    
