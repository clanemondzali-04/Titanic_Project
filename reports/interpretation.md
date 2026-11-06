# Interprétation des résultats du modèle Titanic

## 1. Présentation générale

L'évaluation du modèle de classification Titanic a été réalisée sur un jeu de test indépendant afin de mesurer sa capacité à prédire la survie des passagers.

Les performances ont été évaluées à l'aide de plusieurs métriques : Accuracy, Precision, Recall, F1-score et ROC-AUC.

Les résultats obtenus permettent d'analyser la qualité des prédictions et d'identifier les principales limites du modèle.

## 2. Analyse des métriques

### Accuracy : 76,54 %

L'Accuracy représente la proportion de prédictions correctement réalisées par le modèle.

Avec une Accuracy de 76,54 %, le modèle classe correctement environ trois quarts des passagers du jeu de test.

Cette performance indique que le modèle possède une capacité globale de classification satisfaisante, même si une proportion non négligeable de prédictions reste incorrecte.

### Precision : 75,47 %

La Precision mesure la proportion de passagers réellement survivants parmi ceux que le modèle prédit comme survivants.

Une Precision de 75,47 % signifie qu'environ trois quarts des prédictions positives sont correctes.

Le modèle présente donc une fiabilité relativement satisfaisante lorsqu'il prédit qu'un passager a survécu.

### Recall : 57,97 %

Le Recall mesure la capacité du modèle à identifier les passagers ayant réellement survécu.

Avec un Recall de 57,97 %, le modèle détecte un peu plus de la moitié des survivants réels.

Cette métrique constitue la principale limite du modèle. En effet, 29 survivants ont été incorrectement classés comme décédés.

Le modèle présente donc des difficultés à identifier l'ensemble des passagers survivants.

### F1-score : 65,57 %

Le F1-score représente la moyenne harmonique entre la Precision et le Recall.

Sa valeur de 65,57 % traduit un équilibre modéré entre la fiabilité des prédictions positives et la capacité à détecter les survivants.

Cette valeur est notamment influencée par le Recall relativement faible.

### ROC-AUC : 79,71 %

Le ROC-AUC mesure la capacité du modèle à distinguer les passagers survivants des passagers décédés à différents seuils de décision.

Une valeur de 79,71 % indique que le modèle possède une capacité de discrimination satisfaisante.

Cette métrique complète l'Accuracy en permettant d'évaluer la capacité du modèle à différencier les deux classes indépendamment d'un seuil de décision particulier.

## 3. Analyse de la matrice de confusion

Les résultats montrent que :

- 97 passagers décédés ont été correctement classés.
- 40 passagers survivants ont été correctement identifiés.
- 13 passagers décédés ont été prédits comme survivants.
- 29 passagers survivants ont été prédits comme décédés.

Le modèle reconnaît correctement une proportion importante des passagers décédés.

Cependant, il présente davantage de difficultés dans l'identification des survivants.

Cette observation confirme les résultats obtenus avec le Recall.

## 4. Limites du modèle

Malgré des performances globales satisfaisantes, plusieurs limites peuvent être identifiées.

La principale limite concerne la détection des survivants. Le Recall de 57,97 % montre que le modèle ne parvient pas à identifier tous les passagers ayant survécu.

Par ailleurs, l'Accuracy de 76,54 % indique qu'environ 23,46 % des prédictions sont incorrectes.

Ces résultats montrent que le modèle peut encore être amélioré, notamment en travaillant sur la capacité à identifier les survivants.

## 5. Conclusion générale

L'évaluation du modèle Titanic montre des performances globalement satisfaisantes, avec une Accuracy de 76,54 % et un ROC-AUC de 79,71 %.

Le modèle présente une capacité intéressante à distinguer les passagers survivants des passagers décédés.

Cependant, le Recall de 57,97 % constitue sa principale faiblesse. En effet, le modèle ne détecte qu'environ 58 % des survivants réels, ce qui représente 29 survivants incorrectement classés comme décédés.

Ainsi, bien que le modèle présente une performance globale acceptable, sa capacité à identifier les survivants pourrait être améliorée.

Les prochaines étapes pourraient consister à étudier l'ajustement du seuil de décision, à analyser l'importance des variables et à comparer les performances avec d'autres modèles de classification.

L'objectif serait d'améliorer le Recall tout en conservant un niveau satisfaisant de Precision et d'Accuracy.