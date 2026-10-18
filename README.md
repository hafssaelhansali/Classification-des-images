# Classification d’images : Bateaux et Voitures 

##  Description
Ce projet consiste à classifier des images de bateaux et de voitures en utilisant des algorithmes de Machine Learning classique.

##  Objectifs
- Extraire des caractéristiques des images (couleur, texture et forme).
- Entraîner plusieurs modèles de classification.
- Comparer leurs performances à l’aide de différentes métriques.

##  Technologies utilisées
- Python
- Google Colab
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Scikit-image

##  Modèles utilisés
- SVM (RBF, linéaire, sigmoïde et polynomial)
- K-Nearest Neighbors (KNN)
- Decision Tree

##  Méthodologie
1. Chargement et préparation des images.
2. Extraction des caractéristiques : RGB, histogramme HSV, GLCM, LBP et moments de Hu.
3. Séparation des données en ensembles d’apprentissage et de test.
4. Entraînement et optimisation des modèles.
5. Évaluation et comparaison des résultats.

##  Évaluation
Les modèles sont comparés à l’aide de :
- Accuracy
- Precision
- Recall
- F1-score
- Matrices de confusion

##  Exécution
Le projet est réalisé dans un notebook Jupyter sur Google Colab.

1. Ouvrir le fichier `classification_des_images.ipynb`.
2. Placer le dataset dans Google Drive.
3. Adapter le chemin du dataset dans le notebook.
4. Exécuter les cellules dans l’ordre.

## 
Auteur
Projet académique réalisé dans le cadre du Machine Learning.
