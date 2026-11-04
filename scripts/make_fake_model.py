"""Entraîne un modèle simple et le sauvegarde (modèle provisoire).

Lancer avec : uv run python -m scripts.make_fake_model
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.data import split_data
from src.evaluate import evaluate
from src.predict import save_model


def main():
    X_train, X_test, y_train, y_test = split_data()

    preprocess = ColumnTransformer(
        [
            # numériques : valeurs manquantes -> médiane
            ("num", SimpleImputer(strategy="median"), config.NUMERIC_FEATURES),
            # catégorielles : valeurs manquantes -> plus fréquente, puis OneHot
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train, y_train)
    save_model(pipeline)
    print("Modèle enregistré :", config.MODEL_PATH)
    print(evaluate(pipeline, X_test, y_test))


if __name__ == "__main__":
    main()
