"""
adaptive_model.py — Modèle ML pour la sélection adaptative de questions
Auteur : Moi (ESIEA 3A)

Idée : utiliser les historiques de réponses de l'utilisateur pour lui proposer
des questions au bon niveau — ni trop facile (ennuyeux) ni trop dur (décourageant).

J'ai essayé avec KNN d'abord, ça marchait pas terrible sur des données déséquilibrées,
du coup j'ai switché sur RandomForest. Meilleur résultat en validation croisée.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json

# Chemin vers les données simulées
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/dataset_quiz.csv")

# Profils utilisateurs en mémoire (simple dict pour l'instant)
# TODO: remplacer par une vraie base de données (SQLite ou PostgreSQL)
user_profiles: dict = {}

# Encodeur pour la variable "sujet"
label_encoder = LabelEncoder()

# Modèle global — chargé une fois au démarrage
modele: RandomForestClassifier = None
df_global: pd.DataFrame = None


def charger_modele():
    """
    Charge le dataset et entraîne le modèle RandomForest.
    Appelé au démarrage de l'app.
    """
    global modele, df_global, label_encoder

    try:
        df = pd.read_csv(DATA_PATH)
        df_global = df.copy()

        # Encodage de la colonne 'sujet' (catégorielle → numérique)
        df["sujet_encode"] = label_encoder.fit_transform(df["sujet"])

        # Features pour prédire la difficulté optimale
        X = df[["score", "temps_secondes", "sujet_encode"]]
        y = df["niveau_difficulte"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        modele = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # j'ai testé des valeurs entre 5 et 15, 8 semblait bien
            random_state=42,
        )
        modele.fit(X_train, y_train)

        score_train = modele.score(X_train, y_train)
        score_test = modele.score(X_test, y_test)
        print(f"[MODELE] Accuracy train: {score_train:.3f} | test: {score_test:.3f}")

    except FileNotFoundError:
        print(f"[ERREUR] Dataset introuvable : {DATA_PATH}")
        print("[INFO] Lance d'abord : python app/data/generate_data.py")
        # On crée un modèle vide pour pas crasher l'API
        modele = None


def get_user_profile(user_id: str) -> dict:
    """Retourne le profil d'un utilisateur, le crée s'il n'existe pas."""
    if user_id not in user_profiles:
        user_profiles[user_id] = {
            "user_id": user_id,
            "niveau_actuel": 2,  # on commence en niveau intermédiaire
            "score_total": 0,
            "nb_questions": 0,
            "bonnes_reponses": 0,
            "historique": [],  # liste des (question_id, resultat, temps)
            "sujets_faibles": [],  # sujets où l'utilisateur a du mal
        }
    return user_profiles[user_id]


def update_user_profile(
    user_id: str, question_id: int, score: int, temps_secondes: float, sujet: str
):
    """
    Met à jour le profil utilisateur après une réponse.
    Recalcule le niveau optimal via le modèle ML.
    """
    profil = get_user_profile(user_id)

    profil["nb_questions"] += 1
    profil["score_total"] += score
    if score == 1:
        profil["bonnes_reponses"] += 1

    # Ajout dans l'historique (on garde max 50 dernières entrées pour pas surcharger)
    profil["historique"].append(
        {
            "question_id": question_id,
            "score": score,
            "temps": temps_secondes,
            "sujet": sujet,
        }
    )
    if len(profil["historique"]) > 50:
        profil["historique"] = profil["historique"][-50:]

    # Mise à jour des sujets faibles
    # Si l'utilisateur rate beaucoup dans un sujet, on le note
    historique_sujet = [h for h in profil["historique"] if h["sujet"] == sujet]
    if len(historique_sujet) >= 3:
        taux_reussite_sujet = sum(h["score"] for h in historique_sujet) / len(
            historique_sujet
        )
        if taux_reussite_sujet < 0.4 and sujet not in profil["sujets_faibles"]:
            profil["sujets_faibles"].append(sujet)
        elif taux_reussite_sujet >= 0.6 and sujet in profil["sujets_faibles"]:
            profil["sujets_faibles"].remove(sujet)

    # Prédiction du nouveau niveau optimal via le modèle
    if modele is not None:
        try:
            sujet_encode = label_encoder.transform([sujet])[0]
            features = np.array([[score, temps_secondes, sujet_encode]])
            nouveau_niveau = modele.predict(features)[0]

            # Lissage : on fait une moyenne pondérée entre l'ancien niveau et le nouveau
            # pour éviter des changements trop brutaux
            poids_nouveau = 0.3  # TODO: rendre ça dynamique selon le nb de questions
            niveau_lisse = int(
                round(
                    (1 - poids_nouveau) * profil["niveau_actuel"]
                    + poids_nouveau * nouveau_niveau
                )
            )
            # Clamp entre 1 et 5
            profil["niveau_actuel"] = max(1, min(5, niveau_lisse))

        except Exception as e:
            print(f"[WARN] Erreur prédiction niveau: {e}")
            # Fallback : ajustement manuel basique
            _ajuster_niveau_manuel(profil, score)
    else:
        _ajuster_niveau_manuel(profil, score)


def _ajuster_niveau_manuel(profil: dict, score: int):
    """
    Ajustement de niveau basique sans ML.
    Utilisé si le modèle n'est pas disponible — c'est mon fallback.
    """
    if len(profil["historique"]) >= 3:
        # On regarde les 3 dernières réponses
        recents = profil["historique"][-3:]
        taux = sum(h["score"] for h in recents) / 3

        if taux == 1.0 and profil["niveau_actuel"] < 5:
            profil["niveau_actuel"] += 1  # tout bon → on monte
        elif taux < 0.34 and profil["niveau_actuel"] > 1:
            profil["niveau_actuel"] -= 1  # moins d'1/3 → on descend


def select_question(user_id: str, sujet: str = None) -> dict:
    """
    Sélectionne la prochaine question adaptée au niveau de l'utilisateur.
    Si un sujet est spécifié, on filtre dessus. Sinon on prend au hasard.
    """
    profil = get_user_profile(user_id)
    niveau_cible = profil["niveau_actuel"]

    if df_global is None:
        # Pas de données chargées — on retourne une question hardcodée de secours
        return _question_fallback(niveau_cible)

    # Filtre sur le niveau (on accepte niveau ± 1 pour avoir plus de choix)
    niveaux_acceptes = [
        max(1, niveau_cible - 1),
        niveau_cible,
        min(5, niveau_cible + 1),
    ]
    df_filtre = df_global[df_global["niveau_difficulte"].isin(niveaux_acceptes)]

    # Filtre sur le sujet si demandé
    if sujet and sujet in df_global["sujet"].unique():
        df_filtre = df_filtre[df_filtre["sujet"] == sujet]

    # Priorité aux sujets faibles de l'utilisateur
    if profil["sujets_faibles"] and sujet is None:
        sujet_prioritaire = profil["sujets_faibles"][0]
        df_priorite = df_filtre[df_filtre["sujet"] == sujet_prioritaire]
        if not df_priorite.empty:
            df_filtre = df_priorite

    if df_filtre.empty:
        return _question_fallback(niveau_cible)

    # On choisit une ligne au hasard dans le dataset filtré
    ligne = df_filtre.sample(1).iloc[0]

    return {
        "question_id": int(ligne["question_id"]),
        "sujet": ligne["sujet"],
        "niveau_difficulte": int(ligne["niveau_difficulte"]),
        "enonce": _generer_enonce(ligne["sujet"], int(ligne["niveau_difficulte"])),
        "options": _generer_options(ligne["sujet"], int(ligne["niveau_difficulte"])),
        "bonne_reponse_index": 0,  # dans une vraie app, ce serait stocké en base
    }


def _generer_enonce(sujet: str, niveau: int) -> str:
    """
    Génère un énoncé de question selon le sujet et le niveau.
    C'est simulé ici, dans une vraie app on aurait une vraie BDD de questions.
    TODO: connecter une vraie base de questions (genre SQLite avec 500+ questions)
    """
    enonces = {
        "python": {
            1: "Quelle est la syntaxe pour afficher 'Bonjour' en Python ?",
            2: "Qu'est-ce qu'une list comprehension en Python ?",
            3: "Expliquez la différence entre `*args` et `**kwargs`.",
            4: "Comment fonctionne le décorateur @property en Python ?",
            5: "Qu'est-ce que le GIL (Global Interpreter Lock) et quand pose-t-il problème ?",
        },
        "algo": {
            1: "Quelle est la complexité d'une recherche linéaire ?",
            2: "Expliquez le principe du tri à bulles.",
            3: "Quelle est la différence entre BFS et DFS ?",
            4: "Expliquez la programmation dynamique avec un exemple.",
            5: "Comment fonctionne l'algorithme de Dijkstra ?",
        },
        "math": {
            1: "Qu'est-ce que la dérivée d'une fonction ?",
            2: "Expliquez ce qu'est une matrice diagonale.",
            3: "Qu'est-ce que la décomposition en valeurs propres ?",
            4: "Expliquez la différence entre variance et covariance.",
            5: "Comment fonctionne la descente de gradient stochastique ?",
        },
        "bdd": {
            1: "Qu'est-ce qu'une clé primaire en SQL ?",
            2: "Quelle est la différence entre INNER JOIN et LEFT JOIN ?",
            3: "Qu'est-ce qu'un index en base de données et pourquoi l'utiliser ?",
            4: "Expliquez les propriétés ACID d'une transaction.",
            5: "Comment optimiser une requête SQL lente ?",
        },
    }

    return enonces.get(sujet, {}).get(
        niveau, f"Question de niveau {niveau} en {sujet}."
    )


def _generer_options(sujet: str, niveau: int) -> list:
    """Génère des options de réponse (dont une correcte en position 0)."""
    # TODO: vraies options issues d'une BDD
    return [
        "Bonne réponse (simulée)",
        "Mauvaise réponse A",
        "Mauvaise réponse B",
        "Mauvaise réponse C",
    ]


def _question_fallback(niveau: int) -> dict:
    """Question de secours si le dataset n'est pas dispo."""
    return {
        "question_id": -1,
        "sujet": "python",
        "niveau_difficulte": niveau,
        "enonce": f"[Mode hors-ligne] Question Python de niveau {niveau}",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "bonne_reponse_index": 0,
    }


def get_stats(user_id: str) -> dict:
    """Retourne un résumé des stats de l'utilisateur."""
    profil = get_user_profile(user_id)

    taux_reussite = 0.0
    if profil["nb_questions"] > 0:
        taux_reussite = profil["bonnes_reponses"] / profil["nb_questions"] * 100

    return {
        "user_id": user_id,
        "niveau_actuel": profil["niveau_actuel"],
        "nb_questions_repondues": profil["nb_questions"],
        "taux_reussite": round(taux_reussite, 1),
        "sujets_faibles": profil["sujets_faibles"],
        "progression": _calculer_progression(profil),
    }


def _calculer_progression(profil: dict) -> str:
    """Évalue la progression de l'utilisateur sur les 10 dernières questions."""
    if len(profil["historique"]) < 5:
        return "Pas assez de données"

    recents = profil["historique"][-10:]
    taux = sum(h["score"] for h in recents) / len(recents)

    if taux >= 0.8:
        return "Excellente progression 🚀"
    elif taux >= 0.6:
        return "Bonne progression 👍"
    elif taux >= 0.4:
        return "Progression correcte ➡️"
    else:
        return "Des efforts à fournir 💪"


# Chargement du modèle au démarrage du module
charger_modele()
