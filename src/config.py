"""Constantes partagées par tout le projet."""
from pathlib import Path

# Chemins (ROOT = racine du dépôt)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "titanic.csv"
MODEL_PATH = ROOT / "models" / "titanic_model.joblib"
REPORTS_DIR = ROOT / "reports"

# Colonnes
TARGET = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valeurs possibles des variables catégorielles
SEX_VALUES = ["male", "female"]
EMBARKED_VALUES = ["C", "Q", "S"]
PCLASS_VALUES = [1, 2, 3]

# Découpage train/test
RANDOM_STATE = 42
TEST_SIZE = 0.2
