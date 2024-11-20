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

Les données utilisées ont été récupérées par des méthodes de web-scraping sur le site [ParuVendu](https://www.paruvendu.fr/voiture-occasion/)

Les caractéristiques présentes dans le jeu de données sont les suivantes : 

|Colonnes              | 
| ------------------ |
| Marque          | 
| Modèle          | 
| Llama 3.2 Vision   | 11B        | 7.9GB | `ollama run llama3.2-vision`     |
| Llama 3.2 Vision   | 90B        | 55GB  | `ollama run llama3.2-vision:90b` |
| Llama 3.1          | 8B         | 4.7GB | `ollama run llama3.1`            |
| Llama 3.1          | 70B        | 40GB  | `ollama run llama3.1:70b`        |
| Llama 3.1          | 405B       | 231GB | `ollama run llama3.1:405b`       |
| Phi 3 Mini         | 3.8B       | 2.3GB | `ollama run phi3`                |
