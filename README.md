# Projet de parallélisation de l'algorithme de la Descente du Gradient Stochastique

Auteurs: Etienne Le Naouar, Ghassen Ben Hassine.

### Objectif

Le but de notre projet est de tester les performances de l'algorithme de la Descente du Gradient Stochastique avec la technique de parallélisation tout en utilisant le cas non-parallélisé comme benchmark.

La problématique qu'on cherchera à résoudre est la régression linéaire avec pénalité Ridge.

### Librairie de parallélisation

Dans ce projet, on utilise essentiellement la librarie `multiprocesing`.

### Fichiers auxiliaires

- Le fichier `simulateur_donnees.py` regroupe deux fonctions simulant des données pour le problème de la régression linéaire avec pénalité Ridge.

- Le fichier `SGD_non_para.py` sert à implémenter l'algorithme de la Descente du Gradient Stochastique avec un traitement par lots (mini-batch) et sans parallélisation.

- Le fichier `SGD_para.py` sert à implémenter et entraîner l'algorithme de la Descente du Gradient Stochastique parallélisé avec un traitement par lots (mini-batch).
