"""
generate_data.py — Génération du dataset simulé pour la plateforme
Auteur : Moi (ESIEA 3A)

J'ai généré des données synthétiques pour simuler des historiques de réponses
d'utilisateurs. Dans une vraie plateforme on aurait de vraies données bien sûr,
mais pour prototype c'est largement suffisant.

Lancer ce script une seule fois pour créer le CSV :
    python app/data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

# Seed pour la reproductibilité — important pour que les résultats soient consistants
np.random.seed(42)

# Paramètres du dataset
NB_UTILISATEURS = 200
NB_QUESTIONS_PAR_USER = 10  # en moyenne
NB_TOTAL = 2000  # on cible ~2000 entrées

SUJETS = ["python", "algo", "math", "bdd"]
NIVEAUX = [1, 2, 3, 4, 5]


def generer_dataset(nb_entrees: int = NB_TOTAL) -> pd.DataFrame:
    """
    Génère un dataset simulant des historiques de réponses.

    Logique de simulation :
    - Les utilisateurs ont un "vrai niveau" (1-5) tiré au sort
    - Leur probabilité de réussite dépend de l'écart entre leur niveau et la difficulté
    - Le temps de réponse est corrélé avec la difficulté et le résultat
    """

    lignes = []
    question_counter = 1

    for user_idx in range(NB_UTILISATEURS):
        user_id = f"user_{user_idx:04d}"

        # Niveau "réel" de l'utilisateur (distribution normale centrée sur 2.5)
        niveau_reel_user = np.clip(np.random.normal(2.5, 1.2), 1, 5)

        # Nombre de questions pour cet utilisateur (variable)
        nb_questions_user = np.random.randint(5, 20)

        for _ in range(nb_questions_user):
            sujet = np.random.choice(SUJETS)

            # Niveau de la question (entre 1 et 5)
            # Tendance à poser des questions proches du niveau de l'user
            niveau_question = int(
                np.clip(np.random.normal(niveau_reel_user, 1.0), 1, 5)
            )

            # Probabilité de réussite : dépend de l'écart niveau_user - niveau_question
            ecart = niveau_reel_user - niveau_question
            # Si ecart > 0 : question trop facile → p_reussite haute
            # Si ecart < 0 : question trop dure → p_reussite basse
            p_reussite = 1 / (1 + np.exp(-ecart))  # sigmoid — j'adore cette formule
            p_reussite = np.clip(p_reussite, 0.05, 0.95)

            score = int(np.random.rand() < p_reussite)

            # Temps de réponse : dépend du niveau et du résultat
            # Plus c'est dur, plus ça prend du temps. Si on rate, ça prend plus de temps aussi.
            temps_base = 15 + niveau_question * 8  # entre 23s (niv 1) et 55s (niv 5)
            variation = np.random.normal(0, 5)
            malus_echec = (1 - score) * np.random.uniform(5, 15)  # pénalité si erreur
            temps_secondes = max(5.0, temps_base + variation + malus_echec)

            lignes.append(
                {
                    "user_id": user_id,
                    "question_id": question_counter,
                    "sujet": sujet,
                    "niveau_difficulte": niveau_question,
                    "score": score,
                    "temps_secondes": round(temps_secondes, 1),
                    "niveau_reel_user": round(
                        niveau_reel_user, 2
                    ),  # utile pour analyse
                }
            )

            question_counter += 1

    df = pd.DataFrame(lignes)

    # On s'assure d'avoir ~NB_TOTAL entrées (on complète si besoin)
    if len(df) < nb_entrees:
        # Compléter en dupliquant avec du bruit
        nb_manquants = nb_entrees - len(df)
        df_extra = df.sample(nb_manquants, replace=True).copy()
        df_extra["user_id"] = df_extra["user_id"] + "_bis"
        df_extra["question_id"] = range(
            question_counter, question_counter + nb_manquants
        )
        # Petit bruit sur le temps
        df_extra["temps_secondes"] = (
            (df_extra["temps_secondes"] + np.random.normal(0, 2, nb_manquants))
            .clip(5)
            .round(1)
        )
        df = pd.concat([df, df_extra], ignore_index=True)

    print(f"Dataset généré : {len(df)} entrées")
    print(f"Distribution des scores : {df['score'].value_counts().to_dict()}")
    print(
        f"Distribution des niveaux : \n{df['niveau_difficulte'].value_counts().sort_index()}"
    )

    return df


def sauvegarder_dataset(df: pd.DataFrame, chemin: str):
    """Sauvegarde le dataset en CSV."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    df.to_csv(chemin, index=False)
    print(f"Dataset sauvegardé : {chemin}")


if __name__ == "__main__":
    print("=== Génération du dataset simulé ===")

    df_scores = generer_dataset(NB_TOTAL)

    chemin_csv = os.path.join(os.path.dirname(__file__), "dataset_quiz.csv")
    sauvegarder_dataset(df_scores, chemin_csv)

    # Petit résumé statistique
    print("\n=== Statistiques du dataset ===")
    print(f"Nb utilisateurs uniques : {df_scores['user_id'].nunique()}")
    print(f"Score moyen global : {df_scores['score'].mean():.3f}")
    print(f"Temps moyen de réponse : {df_scores['temps_secondes'].mean():.1f}s")

    # Distribution par sujet
    print("\nTaux de réussite par sujet :")
    print(df_scores.groupby("sujet")["score"].mean().round(3))
