"""Évaluation du modèle."""
import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)
from src.data import split_data

# 1. Chemins
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "titanic_model.joblib"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# 2. Fonction d'évaluation

def evaluate(model, X_test, y_test, model_name="Titanic"):

    """Calcule les métriques de performance."""

    # Prédictions
    y_pred = model.predict(X_test)

    # Probabilités de prédiction
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_proba)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    # Résultats
    metrics = {"model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    # Sauvegarde JSON
    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Affichage de la matrice
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Died", "Survived"]
    )

    disp.plot(cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Courbe ROC
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba
    )

    plt.title("ROC Curve - Titanic")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    return metrics


#Compare le score de validation croisée et le score de test
def compare_scores(cv_accuracy, test_accuracy):

    ecart = abs(cv_accuracy - test_accuracy)
    
    print("\n= CV VS TEST =")

    print(f"CV Accuracy   : {cv_accuracy * 100:.2f}%")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"Écart         : {ecart * 100:.2f}%")

    if ecart < 0.02:
        print("Interpretation : Très bonne stabilité.")

    elif ecart < 0.05:
        print("Interpretation : Stabilité acceptable.")

    else:
        print("Interpretation : Écart important à examiner.")

    


# 3. Exécution

if __name__ == "__main__":

    print("\n- MODEL EVALUATION -")

    # Chargement des données
    X_train, X_test, y_train, y_test = split_data()

    print("Test data loaded:", X_test.shape)

    # Chargement du modèle
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
   # Évaluation
    results = evaluate(
        model,
        X_test,
        y_test
    )

    # Affichage
    print("\n- RESULTS -")

    for name, value in results.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    print("\n- CLASSIFICATION REPORT -")

    print(
        classification_report(
            y_test,
            model.predict(X_test),
            target_names=["Died", "Survived"]
        )
    )

    print("\nResults saved in:", RESULTS_DIR)
    
    # Comparaison CV / Test
    compare_scores(
            cv_accuracy=0.796,
            test_accuracy=results["accuracy"]
        )