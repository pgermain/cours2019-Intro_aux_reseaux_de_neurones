# Auteur: Pascal Germain, 2019
# Matériel du cours: http://chercheurs.lille.inria.fr/pgermain/neurones2019/index.html

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy


class ReseauClassifGenerique:
    """ Classe destinée en encapsuler une architecture de réseau de neurones pour la classification 
    multi-classe. L'apprentissage est effectué par descente de gradient stochastique avec «minibatch»,
    et il est possible de déterminer l'arrêt de l'optimisation par «early stopping».

    Paramètres
    ----------
    modele: Objet contenant l'architecture du réseau de neurones à optimiser. 
    eta, alpha: Parametres de la descente en gradient stochastique (taille du gradient et momentum).
    nb_epoques: Nombre d'époques maximum de la descente en gradient stochastique.
    taille_batch: Nombre d'exemples pour chaque «minibatch».
    fraction_validation: Fraction (entre 0.0 à 1.0) des exemples d'apprentissage à utiliser pour
                         créer un ensemble de validation pour le «early stopping».
                         Par défaut fraction_validation=None et il n'y a pas de «early stopping».
    patience: Paramètre de patience pour le «early stopping».
    seed: Germe du générateur de nombres aléatoires.
    """
    def __init__(self, modele, eta=0.4, alpha=0.1, nb_epoques=10, taille_batch=32, 
                 fraction_validation=0.2, patience=10, seed=None):
        # Initialisation des paramètres
        self.modele = modele
        self.eta = eta
        self.alpha = alpha
        self.nb_epoques = nb_epoques
        self.taille_batch = taille_batch
        self.fraction_validation = fraction_validation
        self.patience = patience
        self.seed = seed
        
        # Ces deux listes serviront à maintenir des statistiques lors de l'optimisation
        self.liste_objectif = list()
        self.liste_erreur_train = list()
        self.liste_erreur_valid = list()
        
    def _trace(self, obj, erreur_train, erreur_valid):
        self.liste_objectif.append(obj.item())    
        self.liste_erreur_train.append(erreur_train.item())
        if self.fraction_validation is not None:
            self.liste_erreur_valid.append(erreur_valid.item()) 
        
    def apprentissage(self, x, y):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)              
        nb_sorties = len(torch.unique(y))
                
        if self.fraction_validation is None:
            # Aucun «early stopping»
            early_stopping = False
            erreur_valid = None
            meilleure_epoque = None
            
            # Toutes les données sont dédiées à l'apprentissage
            train_data = TensorDataset(x, y)
            max_epoques = self.nb_epoques
        else:
            early_stopping = True
            
            # Création de l'ensemble de validation pour le «early stopping»
            nb_valid = int(self.fraction_validation * len(y))
            nb_train = len(y) - nb_valid
            
            train_data = TensorDataset(x[:nb_train], y[:nb_train])
            valid_data = TensorDataset(x[nb_train:], y[nb_train:])
            
            # Initialisation des variables pour le «early stopping»
            meilleure_erreur = 2.
            meilleure_epoque = 0
            max_epoques = self.patience
            
        # Initialisation du problème d'optimisation
        sampler = DataLoader(train_data, batch_size=self.taille_batch, shuffle=True) 
        perte_logistique = nn.NLLLoss()       
        optimizer = torch.optim.SGD(self.modele.parameters(), lr=self.eta, momentum=self.alpha)
           
        # Descente de gradient
        t = 0
        while t < min(max_epoques, self.nb_epoques):
            t += 1
            
            # Une époque correspond à un passage sur toutes les «mini-batch»
            liste_pertes = list()
            for batch_x, batch_y in sampler:
                
                # Propagation avant
                y_pred = self.modele(batch_x)
                perte = perte_logistique(y_pred, batch_y)

                # Rétropropagation
                optimizer.zero_grad()
                perte.backward()
                optimizer.step()
                
                liste_pertes.append(perte.item())
                
            # Pour fin de consultation future, on conserve les statistiques sur la fonction objectif
            perte_moyenne = np.mean(liste_pertes)
            message = f'[{t:3}] perte: {perte_moyenne:.5f}'
            
            # Calcule l'erreur sur l'ensemble d'entraînement
            with torch.no_grad():
                pred_train = self.modele(train_data.tensors[0])
                pred_train = torch.argmax(pred_train, dim=1)
                erreur_train = 1 - torch.mean(pred_train == train_data.tensors[1], dtype=torch.float32)
                message += f' | erreur train: {erreur_train:3f}'            
            
            if early_stopping:
                # Calcule l'erreur sur l'ensemble de validation
                with torch.no_grad():
                    pred_valid = self.modele(valid_data.tensors[0])
                    pred_valid = torch.argmax(pred_valid, dim=1)
                    erreur_valid = 1 - torch.mean(pred_valid == valid_data.tensors[1], dtype=torch.float32)
                    message += f' | erreur valid: {erreur_valid:3f}'
               
                if erreur_valid < meilleure_erreur:
                    # Conserve le meilleur modèle 
                    meilleur_modele = deepcopy(self.modele)
                    meilleure_erreur = erreur_valid
                    meilleure_epoque = t
                    max_epoques = t + self.patience
                    message += f' <-- meilleur modèle à ce jour (max_t={max_epoques})' 
            
            # Fin de l'époque: affiche le message d'état à l'utilisateur avant de passer à l'époque t+1                     
            print(message)
            self._trace(perte_moyenne, erreur_train, erreur_valid)
        
        print('=== Optimisation terminée ===')
        
        # Dans le cas du «early stopping», on retourne à l'état du modèle offrant la meilleure précision en validation  
        if early_stopping:
            self.modele = meilleur_modele
            self.meilleure_epoque = meilleure_epoque
            print(f"Early stopping à l'époque #{meilleure_epoque}, avec erreur de validation de {meilleure_erreur}")
                
    def prediction(self, x):
        x = torch.tensor(x, dtype=torch.float32) # On s'assure que les données sont dans le bon format pytorch       
        
        with torch.no_grad():
            pred = self.modele(x)            # Propagation avant 
            pred = torch.argmax(pred, dim=1) # La prédiction correspond à l'indice du neurone de sortie ayant la valeure maximale
        
        return np.array(pred) # Retourne le résultat en format numpy
