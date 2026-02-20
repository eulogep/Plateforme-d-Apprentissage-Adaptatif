"""
questions.py — Routes FastAPI pour la gestion des quiz adaptatifs
Auteur : Moi (ESIEA 3A)

J'ai séparé les routes dans un fichier à part pour garder main.py propre.
Pydantic pour la validation des données — vraiment pratique, zéro validation manuelle.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import time

# Import du modèle adaptatif
from app.models.adaptive_model import (
    select_question,
    update_user_profile,
    get_stats,
    get_user_profile,
    user_profiles,
)

router = APIRouter()


# ============================================================
# Schémas Pydantic — validation automatique des requêtes/réponses
# ============================================================


class ReponseUtilisateur(BaseModel):
    """Corps de la requête POST /reponse"""

    user_id: str = Field(..., description="Identifiant unique de l'utilisateur")
    question_id: int = Field(..., description="ID de la question répondue")
    reponse_index: int = Field(
        ..., ge=0, le=3, description="Index de la réponse choisie (0-3)"
    )
    sujet: str = Field(..., description="Sujet de la question")
    niveau_difficulte: int = Field(..., ge=1, le=5, description="Niveau de la question")
    temps_secondes: float = Field(
        ..., gt=0, description="Temps pris pour répondre (en secondes)"
    )


class QuestionReponse(BaseModel):
    """Réponse de GET /questions"""

    question_id: int
    sujet: str
    niveau_difficulte: int
    enonce: str
    options: list[str]
    # Note : on n'envoie pas la bonne réponse au client évidemment !


class ResultatReponse(BaseModel):
    """Retour après POST /reponse"""

    correct: bool
    feedback: str
    nouveau_niveau: int
    bonne_reponse_index: int


class StatsUtilisateur(BaseModel):
    """Retour de GET /stats/{user_id}"""

    user_id: str
    niveau_actuel: int
    nb_questions_repondues: int
    taux_reussite: float
    sujets_faibles: list[str]
    progression: str


class ResetConfirmation(BaseModel):
    """Retour de POST /reset/{user_id}"""

    message: str
    user_id: str


# ============================================================
# Endpoints
# ============================================================


@router.get("/questions", response_model=QuestionReponse)
def get_question(
    user_id: str = Query(..., description="ID de l'utilisateur"),
    sujet: Optional[str] = Query(
        None, description="Sujet souhaité (python/algo/math/bdd)"
    ),
):
    """
    Retourne une question adaptée au niveau actuel de l'utilisateur.
    Si le sujet n'est pas précisé, le modèle choisit en priorité les sujets faibles.
    """
    sujets_valides = ["python", "algo", "math", "bdd"]
    if sujet and sujet not in sujets_valides:
        raise HTTPException(
            status_code=400,
            detail=f"Sujet invalide. Valeurs acceptées : {sujets_valides}",
        )

    try:
        question = select_question(user_id, sujet)
        return QuestionReponse(
            question_id=question["question_id"],
            sujet=question["sujet"],
            niveau_difficulte=question["niveau_difficulte"],
            enonce=question["enonce"],
            options=question["options"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la sélection: {str(e)}"
        )


@router.post("/reponse", response_model=ResultatReponse)
def post_reponse(reponse: ReponseUtilisateur):
    """
    Reçoit la réponse d'un utilisateur, vérifie si elle est correcte,
    met à jour son profil et retourne le résultat.

    Note : dans cette version simulée, la bonne réponse est toujours l'index 0.
    TODO: stocker les vraies bonnes réponses en BDD.
    """
    # Validation du sujet
    sujets_valides = ["python", "algo", "math", "bdd"]
    if reponse.sujet not in sujets_valides:
        raise HTTPException(status_code=400, detail="Sujet invalide")

    # Vérification de la réponse (simulée — index 0 = bonne réponse)
    bonne_reponse_index = 0
    est_correct = reponse.reponse_index == bonne_reponse_index
    score = 1 if est_correct else 0

    try:
        # Mise à jour du profil utilisateur via le modèle ML
        update_user_profile(
            user_id=reponse.user_id,
            question_id=reponse.question_id,
            score=score,
            temps_secondes=reponse.temps_secondes,
            sujet=reponse.sujet,
        )

        profil = get_user_profile(reponse.user_id)

        feedback = _generer_feedback(
            est_correct, reponse.temps_secondes, reponse.niveau_difficulte
        )

        return ResultatReponse(
            correct=est_correct,
            feedback=feedback,
            nouveau_niveau=profil["niveau_actuel"],
            bonne_reponse_index=bonne_reponse_index,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du traitement: {str(e)}"
        )


@router.get("/stats/{user_id}", response_model=StatsUtilisateur)
def get_statistiques(user_id: str):
    """
    Retourne les statistiques de progression d'un utilisateur.
    """
    try:
        stats = get_stats(user_id)
        return StatsUtilisateur(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur récupération stats: {str(e)}"
        )


@router.post("/reset/{user_id}", response_model=ResetConfirmation)
def reset_profil(user_id: str):
    """
    Remet à zéro le profil d'un utilisateur.
    Utile pour recommencer à zéro ou pour les tests.
    """
    try:
        if user_id in user_profiles:
            del user_profiles[user_id]
            message = f"Profil de {user_id} réinitialisé avec succès."
        else:
            # L'utilisateur n'existait pas — pas grave, on confirme quand même
            message = f"Profil de {user_id} créé (nouveau profil)."

        # Recréation d'un profil vierge
        get_user_profile(user_id)

        return ResetConfirmation(message=message, user_id=user_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur reset: {str(e)}")


# ============================================================
# Fonctions utilitaires
# ============================================================


def _generer_feedback(correct: bool, temps: float, niveau: int) -> str:
    """Génère un message de feedback personnalisé."""
    if correct:
        if temps < 15:
            return "Excellent ! Réponse rapide et correcte 🚀"
        elif temps < 40:
            return "Très bien ! Bonne réponse 👍"
        else:
            return "Correct ! Essaie d'aller un peu plus vite la prochaine fois ⏱️"
    else:
        if niveau >= 4:
            return "Pas grave, c'était une question difficile. Continue comme ça 💪"
        else:
            return "Ce n'est pas la bonne réponse. Relis le cours sur ce point ! 📚"
