## Classe jouet élaborée dans le cadre d'un TD d'introduction à Python
## Auteur: Pascal Germain (2019)

class vecteur:
    def __init__(self, taille, valeur_initiale=0):
        self.taille = taille
        self.elements = [valeur_initiale] * taille

    def afficher(self):
        print(self.elements)
        
    def copie(self):
        nouveau = vecteur(self.taille)
        nouveau.elements = self.elements.copy()
        return nouveau
    
    def acces(self, index):
        return self.elements[index]
    
    def modif(self, index, valeur):
        self.elements[index] = valeur
    
    def addition(self, valeur):
        self.elements = [x + valeur for x in self.elements]
        
    def norme(self):
        cumul = 0.0
        for x in self.elements:
            cumul += x ** 2
        return cumul ** 0.5      