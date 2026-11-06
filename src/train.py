"""Entraînement : compare 4 modèles en validation croisée et sauvegarde le meilleur."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import split_data
from src.predict import save_model


def build_preprocessor():
    """Prétraitement fait DANS le pipeline, pour que l'API et Streamlit reçoivent des valeurs brutes."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # remplace les âges manquants
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),  # remplace Embarked manquant
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # male/female, C/Q/S -> 0/1
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    """Les 4 modèles et leurs réglages."""
    return {
        "Régression logistique": LogisticRegression(max_iter=1000),
        "Arbre de décision": DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_STATE),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def main():
    # On n'utilise que le train : le test reste intact pour l'évaluation finale
    X_train, _, y_train, _ = split_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    best_name, best_score, best_pipeline = None, -1.0, None
    for name, model in get_models().items():
        pipeline = Pipeline([("preprocessing", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"{name} : accuracy moyenne = {scores.mean():.3f} (ecart-type {scores.std():.3f})")
        if scores.mean() > best_score:
            best_name, best_score, best_pipeline = name, scores.mean(), pipeline

    # Le meilleur modèle est réentraîné sur tout le train, puis sauvegardé
    best_pipeline.fit(X_train, y_train)
    save_model(best_pipeline)
    print(f"Meilleur modèle : {best_name} ({best_score:.3f}). Sauvegardé dans models/titanic_model.joblib")


if __name__ == "__main__":
    main()