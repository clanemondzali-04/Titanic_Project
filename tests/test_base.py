from src.data import split_data
from src.predict import load_model, predict_passenger


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
