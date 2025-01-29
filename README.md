# CarPricePredict : Modele de prédiction de prix pour voiture d'occasion  

## Objectifs :

- Prédire le prix de vente de voiture d'occasion en fonction de leurs caractéristiques en utilisant des modèles de machine learning.
- Utilisation de ce modele pour aider les vendeurs/acheteurs ou les plateformes de vente automobile. 

## Etapes principales : 

- Récupération des données et création du dataset
- Entraînement des différents modèles de machine learning
- Ajustement des hyperparamètres par le biais de validation croisée
- Sélection du modèle sur la base des métriques suivantes : `MAE` et du `coefficient de détermination`. 
- Sauvegarde du modèle sélectionné 
- Création d'un widget pour la prédiction des prix

## Dataset

Les données utilisées ont été récupérées par des méthodes de web-scraping sur le site [ParuVendu](https://www.paruvendu.fr/voiture-occasion/).

Le script de web-scrping est le suivant : [scraping](./scraping.ipynb)

Les caractéristiques présentes dans le jeu de données sont les suivantes : 

|Colonnes| 
| ------------------|
| `Marque` | 
| `Modèle`| 
| `Kilométrage` |
|`Nbre chevaux fiscaux` |
|`Energie` |
| `Nbre places` | 
| `Date mise en circulation`| 
| `Emission Co2` |

La varible à prédire est le `prix`.

| Fichier/Module  | Description  |
|---|---|
| [`scraping.ipynb`](./scraping.ipynb)  | script pour le scraping  |
|  [`cleaning.py`](./cleaning.py) | Module pour le nettoyage de la base |
| [`propre.csv`](./propre.csv)  | Base de données nettoyées |
| [`predictions.py`](./predictions.py)  | Création dataset pour modèles, feature selection, crossvalidation train/test split |
| [`ParuVendu.ipnyb`](./ParuVendu_ML.ipnyb)  | Fichier contenant les travaux |

