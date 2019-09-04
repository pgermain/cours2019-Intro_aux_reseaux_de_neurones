# Introduction aux réseaux de neurones (Automne 2019)

Matériel de cours rédigé par Pascal Germain

Site web du cours: http://chercheurs.lille.inria.fr/pgermain/neurones2019/index.html

### Comment télécharger le contenu de ce répertoire GitHub.

Pour télécharger le contenu de ce répertoire github, il suffit d'éxécuter la commande suivante dans un terminal:
```
git clone git@github.com:pgermain/cours2019-Intro_aux_reseaux_de_neurones.git
```
Cela créera un sous-répertoire `cours2019-Intro_aux_reseaux_de_neurones` dans le répertoire courant de votre ordinateur.
Tant que vous ne modifiez pas localement les fichiers dans ce répertoire (vous pouvez les copier à un autre endroit puis les modifier à votre guise), vous pouvez mettre à jour le contenu comme suit:
1. Accéder au répertoire local:
   ```
   cd cours2019-Intro_aux_reseaux_de_neurones
   ```
2. Télécharger la nouvelle version à partir de GitHub: 
   ```
   git pull
   ```

### Installation de Python

Je conseille à tous d'installer la distribution Python [Anaconda](https://www.anaconda.com/).

* Télécharger la version d'Anaconda avec Python 3.7 pour votre système d'exploitation (Linux, MacOS ou Windows):  
https://www.anaconda.com/download/

* Installer Anaconda en suivant les instructions:
https://docs.anaconda.com/anaconda/install/#detailed-installation-information

* Vous devrez possiblement rédémarrer la session de votre utilisateur pour initialiser les variables d'environnement.

* Au besoin, mettre à jour Anaconda. Par ligne de commande:  
```conda update --all```

### Modes de développement en Python

#### Mode interactif
* Dans le cadre du cours, nous allons développer dans un *carnet [Jupyter](http://jupyter.org/)*. 
Démarrer Jupyter pour commencer:  
```jupyter notebook```

* Pour faire quelques essais rapide, il peut parfois être pratique d'exécuter IPython dans un terminal:  
```ipython```

#### Mode script
Une manière plus conventionelle de programmer en python est d'écrire son code dans un (ou plusieurs) fichier(s) texte(s) et de l'exécuter ensuite à l'aide de l'interpréteur python. Il existe plusieurs environnement de développement pour vous assister dans cette tâche. Par exemple:

* Visual Studio Code: https://code.visualstudio.com/
* PyCharm https://www.jetbrains.com/pycharm/

### Quelques tutoriels suggérés sur Python

* Python et data science: http://www.scipy-lectures.org/
* Scikit-Learn: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
* Tutoriel officiel de Python (avec beaucoup de détail): https://docs.python.org/3/tutorial/index.html
* Un bon survol de concepts simples et avancés: https://github.com/jakevdp/WhirlwindTourOfPython
