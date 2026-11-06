# Titanic Survival Predictor

Une IA qui estime les chances de survie d'un passager du Titanic à partir de sa fiche : classe, sexe, âge, famille à bord, prix du billet, port d'embarquement. Le tout est livré de deux façons : une **application web** pour essayer en quelques clics, et une **API** pour l'intégrer dans d'autres outils.

Projet de groupe réalisé à l'ECES par six personnes, une branche chacune.

## Démo en 2 minutes

```bash
git clone https://github.com/clanemondzali-04/Titanic_Project.git
cd Titanic_Project
uv sync
uv run streamlit run streamlit_app/app.py
```

L'application s'ouvre sur http://localhost:8501 : on renseigne un passager, on clique sur **Prédire**, on obtient « Survivant » ou « Non-survivant » avec la probabilité. Il faut seulement Python 3.12 et [uv](https://docs.astral.sh/uv/).

## Ce que ça donne

Le modèle (un arbre de décision) a été testé sur **179 passagers qu'il n'avait jamais vus** :

| Indicateur | Résultat | En clair |
|---|---|---|
| Exactitude | 77 % | trois prédictions sur quatre sont justes |
| Précision | 75 % | quand il annonce un survivant, il a raison 3 fois sur 4 |
| Décès repérés | 88 % | il identifie presque tous les passagers décédés |
| Survivants repérés (rappel) | 58 % | c'est sa limite : il en manque environ 4 sur 10 |

Pendant l'entraînement, le même modèle atteignait environ 80 % en validation croisée : l'écart avec le test reste modéré, le modèle généralise correctement. Quatre modèles ont été comparés (régression logistique, arbre de décision, forêt aléatoire, KNN) et le meilleur a été retenu. Analyse complète, matrice de confusion et courbe ROC dans [`reports/`](reports/).

## Comment c'est construit

- **Données** : 891 passagers ([`data/titanic.csv`](data/titanic.csv)), explorés dans le notebook [`notebooks/eda.ipynb`](notebooks/eda.ipynb).
- **Modèle** : un pipeline scikit-learn qui intègre la préparation des données (valeurs manquantes, encodage, mise à l'échelle) et le modèle. Il est sauvegardé d'un seul bloc : l'application et l'API reçoivent donc des valeurs brutes, sans retraitement.
- **Application** : Streamlit, formulaire en français.
- **API** : FastAPI, avec validation des entrées.

### Utiliser l'API

```bash
uv run uvicorn api.main:app
```

Documentation interactive sur http://localhost:8000/docs. Exemple de requête `POST /predict` :

```json
{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "embarked": "C"}
```

Réponse : `{"survived": true, "probability": 1.0}`. L'âge est facultatif, les valeurs invalides renvoient une erreur 422, et `GET /health` indique que le modèle est chargé.

## L'équipe

| Qui | Rôle | Livrables | Branche |
|---|---|---|---|
| **Clane MONDZALI** | Cheffe de groupe, modèle | `src/train.py` : comparaison de 4 modèles, sauvegarde du meilleur | `MONDZALI_Clane/features/membre-2-model` |
| **Violette KONDA** | Exploration des données | `notebooks/eda.ipynb` : analyse et justification des choix de préparation | `features/Violette/membre-1-Traitement` |
| **Naura MABIALA** | Évaluation | `src/evaluate.py`, métriques, matrice de confusion, courbe ROC, `reports/interpretation.md` | `features/nauramabiala/membre-3-evaluation` |
| **Exaucé AYAMAR** | Sauvegarde du modèle et tests | test de sauvegarde et de rechargement du modèle | `Exaucé/membre-4/serialisation` |
| **Yann OPENDA** | Initialisation du dépôt, application web | structure du projet, `streamlit_app/app.py` | `features/yann-openda/membre-5-streamlit` |
| **Rich MASSOUKOU** | API | `api/main.py`, `tests/test_api.py` | `features/Rich-Massoukou/membre-6-FastApi` |

Chacun a travaillé sur sa propre branche, puis tout a été réuni dans `main`. Dans l'historique Git, certains commits apparaissent sous des pseudos : KVS (Violette), NH (Naura), exauxoxo (Exaucé), clanemondzali-04 (Clane).

## Pour les développeurs

```bash
uv run python -m src.train      # entraîner les modèles et sauvegarder le meilleur
uv run python -m src.evaluate   # évaluer le modèle sur le jeu de test
uv run pytest                   # lancer les 11 tests
```

| Dossier | Contenu |
|---|---|
| `src/` | code partagé : config, données, entraînement, prédiction, évaluation |
| `models/` | modèle sauvegardé (`titanic_model.joblib`) |
| `api/` | API FastAPI |
| `streamlit_app/` | application Streamlit |
| `notebooks/` | exploration des données |
| `reports/` | métriques et graphiques |
| `tests/` | tests automatiques |
| `docs/` | feuille de route de l'équipe |
| `scripts/` | scripts utilitaires |