"""Chargement et découpage des données."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def split_data():
    """Renvoie X_train, X_test, y_train, y_test (80/20, stratifié)."""
    df = pd.read_csv(config.DATA_PATH)
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,  # garde la même proportion de survivants dans train et test
        random_state=config.RANDOM_STATE,
    )
