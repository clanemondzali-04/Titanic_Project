"""Évaluation du modèle."""
from sklearn.metrics import accuracy_score


def evaluate(model, X_test, y_test):
    """Renvoie un dictionnaire avec seulement l'accuracy."""
    return {"accuracy": accuracy_score(y_test, model.predict(X_test))}
