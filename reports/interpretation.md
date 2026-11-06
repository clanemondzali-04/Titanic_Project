Le modèle de régression logistique présente une Accuracy de 80,45 % sur le jeu de test, ce qui signifie qu'il classe correctement environ huit passagers sur dix.

La Precision de 79,31 % indique que, parmi les passagers prédits comme survivants, près de 79 % ont effectivement survécu.

Le Recall de 66,67 % montre que le modèle parvient à identifier environ deux tiers des passagers ayant réellement survécu.

Le F1-score de 72,44 % traduit un équilibre modéré entre la précision des prédictions positives et la capacité à détecter les survivants.

Le ROC AUC de 84,36 % révèle une bonne capacité du modèle à distinguer les passagers survivants des passagers décédés.

La matrice de confusion montre que le modèle a correctement identifié 98 passagers décédés (TN) et 46 passagers survivants (TP).

Cependant, 12 passagers décédés ont été classés à tort comme survivants (FP), tandis que 23 survivants ont été classés à tort comme décédés (FN).

Ces résultats montrent que le modèle identifie plus facilement les passagers décédés que les survivants.

En validation croisée, la régression logistique avait obtenu une Accuracy de 79,60 %, contre 80,45 % sur le jeu de test.

L'écart entre ces deux performances est de seulement 0,85 point de pourcentage, ce qui suggère une bonne stabilité des prédictions.

Cette proximité doit toutefois être interprétée en tenant compte de la comparabilité des données et des protocoles d'évaluation.

Le principal point fort du modèle réside dans sa performance globale et sa capacité de discrimination, illustrée par un ROC AUC de 84,36 %.

Sa principale limite concerne le Recall, qui indique qu'une partie des survivants n'est pas correctement détectée.

En conclusion, la régression logistique offre des résultats satisfaisants, mais son pouvoir de détection des survivants pourrait être amélioré.