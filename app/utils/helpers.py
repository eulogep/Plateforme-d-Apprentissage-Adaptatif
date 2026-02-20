"""
helpers.py — Fonctions utilitaires diverses
Auteur : Moi (ESIEA 3A)

Des petites fonctions qu'on utilise un peu partout.
J'aurais pu les mettre directement dans les autres fichiers mais
c'est plus propre comme ça (j'essaie de respecter le principe de séparation des responsabilités).
"""

import time
import hashlib
from datetime import datetime


def generer_user_id(nom: str = None) -> str:
    """
    Génère un user_id unique.
    Si un nom est fourni, on le hash pour générer un ID déterministe.
    Sinon, on utilise le timestamp.
    """
    if nom:
        # Hash MD5 du nom (pas pour la sécu, juste pour générer un ID court)
        hash_val = hashlib.md5(nom.encode()).hexdigest()[:8]
        return f"user_{hash_val}"
    else:
        timestamp = int(time.time() * 1000)
        return f"user_{timestamp}"


def formater_duree(secondes: float) -> str:
    """
    Formate une durée en secondes en format lisible.
    Ex: 90.5 → "1min 30s"
    """
    if secondes < 60:
        return f"{int(secondes)}s"
    else:
        minutes = int(secondes // 60)
        secs = int(secondes % 60)
        return f"{minutes}min {secs}s"


def calculer_score_ponderer(historique: list, decay: float = 0.9) -> float:
    """
    Calcule un score pondéré avec décroissance exponentielle.
    Les réponses récentes comptent plus que les anciennes.

    decay=0.9 → la question d'avant vaut 90% de la dernière.

    J'ai appris ce concept dans le cours de ML de M. Lambert — c'est une
    forme simple de weighted average avec "forgetting factor".
    """
    if not historique:
        return 0.0

    score_total = 0.0
    poids_total = 0.0
    poids = 1.0

    # On parcourt de la plus récente à la plus ancienne
    for entree in reversed(historique):
        score_total += entree.get("score", 0) * poids
        poids_total += poids
        poids *= decay

    return score_total / poids_total if poids_total > 0 else 0.0


def niveau_vers_label(niveau: int) -> str:
    """Convertit un niveau numérique (1-5) en label lisible."""
    labels = {1: "Débutant", 2: "Intermédiaire", 3: "Avancé", 4: "Expert", 5: "Maître"}
    return labels.get(niveau, "Inconnu")


def timestamp_formaté() -> str:
    """Retourne le timestamp actuel formaté."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Petit test rapide si on lance le fichier directement
if __name__ == "__main__":
    print(generer_user_id("Alice"))
    print(formater_duree(125))
    print(niveau_vers_label(3))

    historique_test = [
        {"score": 1},
        {"score": 0},
        {"score": 1},
        {"score": 1},
        {"score": 0},
    ]
    print(f"Score pondéré : {calculer_score_ponderer(historique_test):.3f}")
