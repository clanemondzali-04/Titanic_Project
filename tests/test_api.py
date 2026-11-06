import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    # Évite de charger un vrai modèle pendant les tests unitaires.
    monkeypatch.setattr(main, "load_model", lambda: object())
    monkeypatch.setattr(
        main,
        "predict_passenger",
        lambda model, passenger: {"survived": True, "probability": 0.91},
    )
    with TestClient(main.app) as test_client:
        yield test_client


def valid_passenger():
    return {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 80.0,
        "embarked": "S",
    }


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=valid_passenger())
    assert response.status_code == 200
    assert response.json() == {"survived": True, "probability": 0.91}


def test_predict_accepts_missing_age(client):
    passenger = valid_passenger()
    passenger["age"] = None
    response = client.post("/predict", json=passenger)
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [("sex", "unknown"), ("embarked", "X"), ("pclass", 4)],
)
def test_predict_rejects_invalid_categories(client, field, value):
    passenger = valid_passenger()
    passenger[field] = value
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_predict_rejects_missing_required_field(client):
    passenger = valid_passenger()
    del passenger["fare"]
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422


def test_root_and_health(client):
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
