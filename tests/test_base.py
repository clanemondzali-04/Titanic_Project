from src import config
from src.predict import save_model
from src.data import split_data
from src.predict import load_model, predict_passenger
import joblib

def test_split_total_891_lignes():
    X_train, X_test, y_train, y_test = split_data()
    assert len(X_train) + len(X_test) == 891
    assert len(y_train) + len(y_test) == 891


def test_modele_se_recharge():
    assert load_model() is not None


def test_predict_passenger_probabilite():
    passenger = {
        "pclass": 3, "sex": "male", "age": None,  # âge manquant accepté
        "sibsp": 0, "parch": 0, "fare": 8.05, "embarked": "S",
    }
    result = predict_passenger(load_model(), passenger)
    assert 0 <= result["probability"] <= 1
    assert isinstance(result["survived"], bool)

def test_sauvegarde_rechargement_predictions_identiques(tmp_path, monkeypatch):
    model = load_model()

    # Créer un chemin temporaire pour ne pas toucher au vrai modèle
    chemin_temporaire = tmp_path / "modele_test.joblib"
    monkeypatch.setattr(config, "MODEL_PATH", chemin_temporaire)

    passenger = {
        "pclass": 3,
        "sex": "male",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S",
    }

    prediction_avant = predict_passenger(model, passenger)

    # Utiliser nos fonctions de sauvegarde et de chargement
    save_model(model)
    modele_recharge = load_model()

    prediction_apres = predict_passenger(modele_recharge, passenger)

    assert prediction_avant == prediction_apres

def test_sauvegarde_rechargement_predictions_identiques(tmp_path, monkeypatch):
    # Charger le modèle original
    model = load_model()

    # Remplacer temporairement le chemin du modèle
    chemin_temporaire = tmp_path / "modele_test.joblib"
    monkeypatch.setattr(config, "MODEL_PATH", chemin_temporaire)

    # Faire une première prédiction
    passenger = {
        "pclass": 3,
        "sex": "male",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S",
    }

    prediction_avant = predict_passenger(model, passenger)

    # Sauvegarder et recharger avec nos propres fonctions
    save_model(model)
    modele_recharge = load_model()

    prediction_apres = predict_passenger(modele_recharge, passenger)

    assert prediction_avant == prediction_apres