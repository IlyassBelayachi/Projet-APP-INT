# Semaine 2 : entraîner, évaluer et accepter un modèle de sentiment

## 1. Mes chiffres

Modèle : TF-IDF (1 et 2 grammes) + régression logistique. Seuil technique de `train.py` : F1 macro ≥ 0,85. Seuil métier de la fiche de cadrage du cas 3 : rappel sur la classe négative ≥ 0,90.

| Mesure | Démonstration (290 avis) | Allociné (20 000 avis) |
| --- | --- | --- |
| Avis d'entraînement / de test | 232 / 58 | 16 000 / 4 000 |
| Vocabulaire | 316 termes | 149 529 termes |
| F1 macro | 0,879 | 0,9205 |
| Précision / rappel, classe négative | 0,844 / 0,931 | 0,931 / 0,911 |
| Rappel sur la classe négative | 0,931 (27 sur 29) | 0,9113 (1 860 sur 2 041) |
| Erreurs sur le jeu de test | 7 sur 58 | 318 sur 4 000 |
| Négatifs classés positifs (faux négatifs) | 2 | 181 |
| Positifs classés négatifs | 5 | 137 |

Matrices de confusion (lignes = vérité, colonnes = prédiction) :

- Démonstration : négatif 27 / 2, positif 5 / 24.
- Allociné : négatif 1 860 / 181, positif 137 / 1 822.

**Écart avec l'énoncé.** L'énoncé annonce, pour la démonstration, un F1 macro de 0,8961, un rappel négatif de 0,9655 et 6 erreurs, et pour Allociné un F1 macro de 0,9020, un rappel négatif de 0,8905 et 392 erreurs. Mes valeurs diffèrent (0,879 / 0,931 / 7 erreurs, et 0,9205 / 0,9113 / 318 erreurs). Les chiffres semblent venir d'une version différente du kit fourni. Je les signale sans avoir modifié aucun seuil. Mon environnement : Python 3.12.4, scikit-learn 1.5.2.

## 2. Lecture des trois erreurs affichées par `evaluate.py`

### Jeu de démonstration

1. « Taille trop petite et retour payant au déballage, sauf que ensuite produit conforme… » (vérité : positif, prédit : négatif)
2. « Commande jamais reçue et remboursement en attente au début, sauf que ensuite je recommande » (vérité : positif, prédit : négatif)
3. « Service impeccable et livraison en avance sur le papier, sauf que ensuite produit cassé… » (vérité : négatif, prédit : positif)

**Ce qu'ils ont en commun.** Ce sont tous des avis mitigés construits de la même façon : une première partie d'un sentiment, un retournement (« sauf que ensuite »), puis une seconde partie du sentiment opposé. L'étiquette suit la conclusion.

**Pourquoi le modèle se trompe.** Le TF-IDF compte des mots et des paires de mots, sans comprendre la structure de la phrase. « Bon puis mauvais » et « mauvais puis bon » donnent à peu près le même vecteur, alors que le sens est opposé : le modèle ne sait pas que « sauf que ensuite » fait basculer le sens. Une partie des erreurs se retrouve ainsi dans les deux sens (2 faux négatifs, 5 faux positifs). Pour les corriger, il faudrait un modèle qui tient compte de l'ordre (embeddings de phrase ou modèle de langue), ou davantage d'avis mitigés étiquetés d'après leur conclusion.

### Jeu Allociné

**Analyse des trois erreurs après lecture des avis complets :**

1. Critique négative prédite positive : l'avis commençant par « Grammaire cinématographique rudimentaire » critique principalement la réalisation, les images et l'interprétation. Le texte contient toutefois de longues descriptions et quelques appréciations favorables. Le modèle peut accorder trop de poids à certains termes sans restituer le jugement global.

2. Critique positive prédite négative : l'avis commençant par « Bravo à Yasmina Reza » est court et clairement élogieux. Cette erreur peut provenir d'une représentation insuffisante de certaines expressions positives dans le vocabulaire appris ou de leurs coefficients statistiques.

3. Critique positive prédite négative : l'avis sur Clint Eastwood est globalement positif, mais contient l'expression « la trahissant légèrement ». Le modèle peut attribuer un poids négatif au terme « trahissant » sans prendre suffisamment en compte la nuance de la phrase.

Ces exemples illustrent les limites du TF-IDF et de la régression logistique dans l'interprétation du contexte sémantique. Les explications restent des hypothèses qui nécessiteraient une analyse des coefficients du modèle pour être confirmées.

## 3. Décision : le modèle est-il accepté, et pour quel usage ?

| Seuil | Démonstration | Allociné |
| --- | --- | --- |
| F1 macro ≥ 0,85 (technique) | 0,879 : passé | 0,9205 : passé |
| Rappel négatif ≥ 0,90 (cas 3) | 0,931 : passé | 0,9113 : passé |

Sur le papier, les deux modèles passent les deux seuils. Je ne m'en tiens pourtant pas à ce constat, pour trois raisons :

- **Démonstration : mesure fragile.** Il n'y a que 29 avis négatifs en test. Un seul négatif de plus mal classé ferait passer le rappel à environ 0,897, sous le seuil. Les données sont aussi synthétiques.
- **Allociné : mesure solide, mais pour un autre domaine.** 4 000 avis rendent le chiffre crédible, mais ce sont des critiques de films, pas des avis du site e-commerce du cas 3. La marge sur le seuil est faible (0,9113 contre 0,90).
- **Test hors domaine.** Voir la rubrique 4 : le modèle Allociné se trompe sur un avis de livraison simple.

**Décision.** J'accepte le modèle comme **composant technique de la chaîne** (c'est lui que l'API de la semaine 3 chargera, et il sera ré-entraîné de façon reproductible avec `python src/train.py`). Je **ne l'accepte pas encore pour un usage en production** sur les avis du site : cela exigerait de le mesurer sur un échantillon d'avis réels du site, annotés à la main, et de vérifier que le rappel négatif reste au moins égal à 0,90 sur ces avis. Avant cela, les avis à faible confiance doivent être relus par un agent.

## 4. Limites constatées

- **Essai hors domaine.** Avec le modèle Allociné, « Livraison rapide, produit conforme, tres satisfait » sort **négatif avec 50 % de confiance** (alors que l'avis est positif), et « Produit casse et service injoignable, a fuir » sort négatif à 75 % (correct). Avec le modèle de démonstration, les mêmes avis sortent positif à 71 % et négatif à 61 %. Le vocabulaire du commerce est absent du corpus Allociné ; une confiance proche de 50 % est le signal d'alerte.
- **Un score ne vaut que pour des données de même nature** que celles de l'entraînement et du test.
- **Ordre des mots ignoré** : les avis mitigés et les retournements (« sauf que ensuite ») sont mal classés.
- **Mesure sur 58 avis** : trop petite pour décider seule, d'où l'intérêt du jeu Allociné.
- **Avertissement `OptimizeWarning: Unknown solver options: iprint`** à l'entraînement (scikit-learn et scipy) : sans effet sur les résultats.
- **Version de Python** :  entraînement et évaluation également vérifiés avec Python 3.11.9 sur Windows, avec 42 tests automatiques réussis. L'environnement initial du rapport utilisait Python 3.12.4.
