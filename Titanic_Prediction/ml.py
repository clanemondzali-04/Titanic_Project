import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Chemin du dossier contenant les données
chemin = Path("C:/Users/hp/Documents/L3_projets/Projet_Docker/BD")

# Chargement des données prétraitées
titanic_train = pd.read_csv(chemin / "train_prepared.csv")

# Séparation des variables explicatives et de la variable cible
X = titanic_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

y = titanic_train['Survived']

# Vérification
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dimensions de X_train :", X_train.shape)
print("Dimensions de X_test :", X_test.shape)

print("Dimensions de y_train :", y_train.shape)
print("Dimensions de y_test :", y_test.shape)

#Algorithme de classification pour prédire la survie des passagers du Titanic
# 1. Régression logistique

logistic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Entraînement du modèle
logistic_model.fit(X_train, y_train)

print("Régression logistique entraînée avec succès.")

# 2. Arbre de décision

tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

# Entraînement du modèle
tree_model.fit(X_train, y_train)

print("Arbre de décision entraîné avec succès.")

# 3. Forêt aléatoire

forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Entraînement du modèle
forest_model.fit(X_train, y_train)

print("Forêt aléatoire entraînée avec succès.")

# 4. K plus proches voisins

knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

# Entraînement du modèle
knn_model.fit(X_train, y_train)

print("KNN entraîné avec succès.")