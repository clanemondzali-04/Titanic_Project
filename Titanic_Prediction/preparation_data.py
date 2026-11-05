import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path #Path permet de manipuler les chemins des fichiers et des dossiers.

#Importation des données et séparation des variables
##chemin = Path(__file__).resolve().parent
chemin = Path("C:\\Users\\hp\\Documents\\L3_projets\\Projet_Docker\\BD")
titanic_train = pd.read_csv(chemin / "train.csv")
titanic_test=pd.read_csv(chemin / "test.csv")

titanic_train=titanic_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] #on sélectionne les variables
titanic_test=titanic_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] #on sélectionne les variables

# Traitement des valeurs manquantes

# Remplacement des âges manquants par la médiane
age_median = titanic_train['Age'].median()

titanic_train['Age'] = titanic_train['Age'].fillna(age_median)
titanic_test['Age'] = titanic_test['Age'].fillna(age_median)

# Remplacement des ports d'embarquement manquants par le mode
embarked_mode = titanic_train['Embarked'].mode()[0]

titanic_train['Embarked'] = titanic_train['Embarked'].fillna(embarked_mode)
titanic_test['Embarked'] = titanic_test['Embarked'].fillna(embarked_mode)

#Modification de l'encodage et conversion des variables catégorielles en numériques   
titanic_train['Sex']=titanic_train['Sex'].replace({'male': 0, 'female': 1})
titanic_test['Sex']=titanic_test['Sex'].replace({'male': 0, 'female': 1})
titanic_train['Sex'] = titanic_train['Sex'].astype(int)
titanic_test['Sex'] = titanic_test['Sex'].astype(int)

titanic_train['Age'] = np.floor(titanic_train['Age'] + 0.5).astype('Int64')
titanic_test['Age'] = np.floor(titanic_test['Age'] + 0.5).astype('Int64')
titanic_train['Embarked'] = titanic_train['Embarked'].replace({
    'S': 0,
    'C': 1,
    'Q': 2
})
titanic_test['Embarked'] = titanic_test['Embarked'].replace({
    'S': 0,
    'C': 1,
    'Q': 2
})
titanic_train['Embarked'] = titanic_train['Embarked'].astype('Int64')
titanic_test['Embarked'] = titanic_test['Embarked'].astype('Int64')

print(titanic_train.head())
print(titanic_test.head())

#Vérification des types de données, des valeurs manquantes et de la tailles des datasets
print(titanic_train.dtypes)
print("\nValeurs manquantes dans train :")
print(titanic_train.isnull().sum())

print("\nValeurs manquantes dans test :")
print(titanic_test.isnull().sum())

print("\nDimensions de train :", titanic_train.shape)
print("Dimensions de test :", titanic_test.shape)

# Enregistrement des données prétraitées
titanic_train.to_csv(chemin / "train_prepared.csv", index=False)
titanic_test.to_csv(chemin / "test_prepared.csv", index=False)