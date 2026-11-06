# Feuille de route interne

## Règles communes
- Chaque membre travaille uniquement sur sa branche.
- Personne ne lance `uv add` : les dépendances sont déjà installées. Besoin d'une nouvelle dépendance : on le demande à celui qui a initialisé le dépôt.
- Conflit sur `uv.lock` : garder la version de main, puis lancer `uv lock`.
- Contrat commun : `src/config.py` (constantes) et les fonctions `split_data()`, `save_model()`, `load_model()`, `predict_passenger()`, `evaluate()`. On ne change jamais leurs noms ni leurs paramètres.
- `predict_passenger(model, passenger)` reçoit un dictionnaire aux clés en minuscules (`pclass, sex, age, sibsp, parch, fare, embarked`) et renvoie `{"survived": bool, "probability": float}`. L'âge peut être absent.
- Le modèle sauvegardé est le pipeline complet (prétraitement + classifieur), pas seulement le classifieur.
- Seul le Membre 2 commite `models/titanic_model.joblib`. Les autres ne modifient pas ce fichier : ne pas relancer `scripts/make_fake_model.py` pour le commiter.

## Membre 1 : Données
- Il reçoit : `data/titanic.csv` (891 lignes), `src/config.py`, `split_data()` de départ.
- Il livre : `src/data.py` solide, `notebooks/exploration.ipynb`, graphiques dans `reports/`.
- En attendant : tout est déjà disponible, il travaille sur le vrai fichier.
- Terminé quand : `split_data()` fait une séparation 80/20 stratifiée avant tout prétraitement, et les graphiques sont dans `reports/`.

## Membre 2 : Modèle (chemin critique)
- Il reçoit : `split_data()` et `src/config.py`.
- Il livre : `src/train.py` avec le pipeline (numériques : médiane puis standardisation ; Sex et Embarked : valeur la plus fréquente puis OneHot), 3 modèles comparés (régression logistique, arbre de décision, Random Forest) en validation croisée à 5 plis sur le train uniquement, sauvegarde du meilleur avec `save_model()`.
- En attendant : le faux modèle déjà présent dans le dépôt (`scripts/make_fake_model.py`).
- Terminé quand : `uv run python -m src.train` produit `models/titanic_model.joblib` et le jeu de test n'a pas été utilisé pour choisir.

## Membre 3 : Évaluation
- Il reçoit : `split_data()`, le faux modèle déjà présent dans le dépôt, `evaluate()` de départ (accuracy seulement).
- Il livre : `src/evaluate.py` complet (accuracy, precision, recall, F1, matrice de confusion) et ses graphiques dans `reports/`.
- En attendant : il teste sur le faux modèle.
- Terminé quand : l'évaluation tourne sur le vrai modèle, une seule fois sur le jeu de test, et les résultats sont dans `reports/`.

## Membre 4 : Joblib, tests, documentation
- Il reçoit : `save_model()`, `load_model()`, `predict_passenger()` de départ et le faux modèle déjà présent dans le dépôt.
- Il livre : versions solides de ces trois fonctions, tests dans `tests/`, `INSTALL.md`.
- En attendant : il teste sur le faux modèle.
- Terminé quand : un modèle sauvegardé puis rechargé donne les mêmes prédictions, et `INSTALL.md` permet à un débutant d'installer et lancer le projet.
- Attention : ses fonctions sont utilisées par d'autres membres, donc signatures figées.

## Membre 5 : Streamlit
- Il reçoit : `load_model()`, `predict_passenger()`, les valeurs autorisées de `src/config.py`, le faux modèle déjà présent dans le dépôt.
- Il livre : `streamlit_app/app.py` avec formulaire, bouton Prédire, affichage du résultat et de la probabilité.
- En attendant : il construit tout avec le faux modèle.
- Terminé quand : `uv run streamlit run streamlit_app/app.py` fonctionne avec le vrai modèle.

## Membre 6 : FastAPI
- Il reçoit : `load_model()`, `predict_passenger()`, les valeurs autorisées de `src/config.py`, le faux modèle déjà présent dans le dépôt.
- Il livre : `api/main.py` avec `GET /`, `GET /health`, `POST /predict` (entrées validées).
- En attendant : il construit tout avec le faux modèle.
- Terminé quand : `uv run uvicorn api.main:app --reload` fonctionne, `/predict` renvoie du JSON, et un test avec `httpx` passe.

## Relectures croisées
M3 relit M1 et M2, M1 relit M3, M2 relit M4, M5 relit M6, M6 relit M5.

## Calendrier
- Celui qui a initialisé le dépôt fusionne sur main dans l'ordre : données, modèle, Joblib, évaluation, API, Streamlit.
- Gel du code : vendredi soir. Rendu : samedi 7 novembre.
- Plan B si le Membre 2 est en retard : une régression logistique simple dans le pipeline.
