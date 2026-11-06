"""Application Streamlit : prédire la survie d'un passager du Titanic."""
import sys
from pathlib import Path

# Ajoute la racine du projet au chemin Python (pour importer "src")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config
from src.predict import load_model, predict_passenger

# Libellés affichés -> valeurs envoyées au modèle (clés = config)
LIBELLES_SEXE = {"male": "Homme", "female": "Femme"}
LIBELLES_PORT = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


@st.cache_resource
def charger_modele():
    """Charge le modèle une seule fois (mis en cache)."""
    return load_model()


st.title("Survie sur le Titanic")
st.write("Renseigne les informations d'un passager pour estimer sa probabilité de survie.")

# Chargement du modèle : message clair si le fichier est absent
try:
    model = charger_modele()
except FileNotFoundError:
    st.error(
        f"Modèle introuvable : {config.MODEL_PATH}. "
        "Génère-le avec `uv run python -m scripts.make_fake_model`."
    )
    st.stop()

# Champs du formulaire
pclass = st.selectbox("Classe", config.PCLASS_VALUES)
sex = st.selectbox("Sexe", config.SEX_VALUES, format_func=LIBELLES_SEXE.get)

age_inconnu = st.checkbox("Âge inconnu")
age_saisi = st.number_input(
    "Âge", min_value=0.0, max_value=100.0, value=30.0, step=1.0,
    disabled=age_inconnu,  # grisé si l'âge est inconnu
)

sibsp = st.number_input("Frères/sœurs/conjoint à bord", min_value=0, value=0, step=1)
parch = st.number_input("Parents/enfants à bord", min_value=0, value=0, step=1)
fare = st.number_input("Prix du billet", min_value=0.0, value=32.0, step=1.0)
embarked = st.selectbox(
    "Port d'embarquement", config.EMBARKED_VALUES, format_func=LIBELLES_PORT.get
)

if st.button("Prédire"):
    passager = {
        "pclass": pclass,
        "sex": sex,
        "age": None if age_inconnu else age_saisi,  # None = âge manquant
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": embarked,
    }
    resultat = predict_passenger(model, passager)

    if resultat["survived"]:
        st.success("Survivant")
    else:
        st.error("Non-survivant")

    proba = resultat["probability"]
    st.metric("Probabilité de survie", f"{proba:.1%}")
    st.progress(proba)
