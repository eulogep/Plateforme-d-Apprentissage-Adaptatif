"""
main.py — Point d'entrée de l'API FastAPI
Auteur : Moi (ESIEA 3A)
Date : Février 2026

J'ai choisi FastAPI parce que c'est ultra rapide à mettre en place
et la doc auto avec Swagger c'est vraiment pratique pour tester sans Postman.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import de mes routes custom
from app.routes.questions import router as questions_router

app = FastAPI(
    title="Plateforme d'Apprentissage Adaptatif",
    description="API pour gérer les quiz adaptatifs selon le niveau de l'utilisateur",
    version="1.0.0"
)

# CORS — nécessaire si je veux connecter un front React un jour
# TODO: restreindre les origines en prod, là c'est trop permissif
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du router principal
app.include_router(questions_router, prefix="/api")


@app.get("/")
def root():
    """Route de santé pour vérifier que l'API tourne."""
    return {"message": "API Apprentissage Adaptatif — v1.0", "status": "ok"}


@app.on_event("startup")
async def startup_event():
    """
    Au démarrage, on charge le modèle ML en mémoire.
    J'ai mis ça ici pour éviter de recharger le modèle à chaque requête
    — j'ai galéré sur ça un moment avant de comprendre comment fonctionne le lifecycle FastAPI.
    """
    print("[INFO] Démarrage de l'API...")
    print("[INFO] Chargement du modèle adaptatif...")
    # Le modèle est chargé dans adaptive_model.py directement (voir là-bas)
    print("[INFO] API prête !")


# Pour lancer en local : uvicorn app.main:app --reload
