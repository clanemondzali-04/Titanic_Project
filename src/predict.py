"""Sauvegarde, chargement du modèle et prédiction pour un passager."""
import joblib
import pandas as pd

from src import config


def save_model(pipeline):
    """Enregistre le pipeline dans models/titanic_model.joblib."""
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)


def load_model():
    """Recharge le pipeline sauvegardé."""
    return joblib.load(config.MODEL_PATH)


def predict_passenger(model, passenger):
    """passenger : dict aux clés pclass, sex, age, sibsp, parch, fare, embarked.

    Renvoie {"survived": bool, "probability": float}.
    L'âge peut être None (il sera imputé par le pipeline).
    """
    row = pd.DataFrame(
        [
            {
                "Pclass": passenger["pclass"],
                "Sex": passenger["sex"],
                # None -> NaN, pour que l'imputation fonctionne
                "Age": float("nan") if passenger.get("age") is None else passenger["age"],
                "SibSp": passenger["sibsp"],
                "Parch": passenger["parch"],
                "Fare": passenger["fare"],
                "Embarked": passenger["embarked"],
            }
        ]
    )[config.FEATURES]  # même ordre de colonnes que pendant l'entraînement
    proba = float(model.predict_proba(row)[0, 1])
    return {"survived": proba >= 0.5, "probability": proba}
