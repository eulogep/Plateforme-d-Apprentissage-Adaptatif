<div align="center">

> Projet personnel développé dans le cadre de ma 3ème année à l'ESIEA.  
> Une API qui adapte les quiz au niveau de l'utilisateur grâce au Machine Learning.

</div>

---

## 🎯 Pourquoi ce projet ?

Honnêtement, ce projet est né d'un problème très concret : quand je révisais avec des quiz classiques, je perdais du temps sur des questions trop faciles **ou** je me décourageais sur des questions bien trop dures.

L'idée était simple : **et si le quiz s'adaptait à moi ?**

J'avais déjà vu des systèmes similaires dans Duolingo ou Khan Academy, mais je ne comprenais pas vraiment le mécanisme. Ce projet m'a permis de creuser les **modèles adaptatifs** et d'appliquer mon cours de ML en pratique.

> 💡 **Résultat :** +15% de progression simulée par rapport à un quiz séquentiel classique.

---

## ⚙️ Architecture du projet

```
learning-platform/
├── 📁 app/
│   ├── main.py                  # FastAPI entrypoint + CORS + lifecycle
│   ├── 📁 models/
│   │   └── adaptive_model.py    # RandomForest + gestion profils utilisateurs
│   ├── 📁 routes/
│   │   └── questions.py         # 4 endpoints REST + schémas Pydantic
│   ├── 📁 data/
│   │   └── generate_data.py     # ~2000 entrées simulées (sigmoid prob)
│   └── 📁 utils/
│       └── helpers.py           # Score pondéré, formatage, utilitaires
├── 📁 notebooks/
│   └── exploration.ipynb        # EDA + entraînement + simulation comparative
├── requirements.txt
└── README.md
```

---

## 🚀 Lancer le projet

### 1️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2️⃣ Générer le dataset simulé

```bash
python app/data/generate_data.py
```

> Crée `app/data/dataset_quiz.csv` avec ~2000 historiques de réponses synthétiques.

### 3️⃣ Démarrer l'API

```bash
uvicorn app.main:app --reload
```

L'API est disponible sur **`http://localhost:8000`**

| Interface | URL |
|-----------|-----|
| 📖 Swagger UI (interactif) | `http://localhost:8000/docs` |
| 📚 ReDoc | `http://localhost:8000/redoc` |

---

## 🔌 Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/api/questions?user_id=xxx` | Retourne une question adaptée au niveau |
| `POST` | `/api/reponse` | Envoie une réponse, met à jour le profil |
| `GET` | `/api/stats/{user_id}` | Stats de progression de l'utilisateur |
| `POST` | `/api/reset/{user_id}` | Remet le profil à zéro |

### Exemple rapide

```bash
# Obtenir une question Python pour user_001
curl "http://localhost:8000/api/questions?user_id=user_001&sujet=python"

# Envoyer une réponse
curl -X POST "http://localhost:8000/api/reponse" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "question_id": 42,
    "reponse_index": 0,
    "sujet": "python",
    "niveau_difficulte": 2,
    "temps_secondes": 25.5
  }'

# Voir les stats
curl "http://localhost:8000/api/stats/user_001"
```

---

## 🤖 Comment fonctionne le modèle adaptatif ?

```
Réponse utilisateur
       │
       ▼
┌─────────────────┐      ┌──────────────────────┐
│  score + temps  │ ───▶ │   RandomForest (ML)   │
│  + sujet        │      │   predict difficulty  │
└─────────────────┘      └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Lissage du niveau   │
                         │  (avg pondéré 70/30) │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Sélection question  │
                         │  niveau ± 1 du user  │
                         └──────────────────────┘
```

- **Dataset** : ~2000 historiques simulés avec probabilité de réussite modélisée par une fonction **sigmoid** (inspiré de la Item Response Theory)
- **Modèle** : `RandomForestClassifier` (100 arbres, max_depth=8) — accuracy ~78% en test
- **Anti-oscillation** : lissage exponentiel entre l'ancien niveau et la prédiction (70/30)
- **Sujets faibles** : détection automatique si taux de réussite < 40% sur un sujet

---

## 📊 Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy train | ~92% |
| Accuracy test | ~78% |
| Validation croisée (5-fold) | 76% ± 2% |
| Gain de progression vs séquentiel | **+15%** |

> Les résultats sont obtenus sur des données simulées. Sur de vraies données utilisateurs, les performances seront différentes — c'est la prochaine étape.

---

## 😅 Difficultés rencontrées

**Comprendre les modèles adaptatifs** — J'ai lu des papiers sur l'IRT (*Item Response Theory*) mais c'est très matheux. J'ai opté pour une approche plus pragmatique avec RandomForest, qui donne de bons résultats sans modélisation probabiliste complexe.

**Le lissage du niveau** — Au départ, le niveau changeait brutalement à chaque réponse. J'ai ajouté un facteur de lissage pour éviter les oscillations (genre niveau 5 → niveau 1 en deux mauvaises réponses 😅).

**Les imports relatifs FastAPI** — La structure `app/` avec `__init__.py` m'a pris du temps à comprendre. J'ai galéré sur ça.

**Les données synthétiques** — Générer des données qui ressemblent à de vraies données comportementales c'est plus complexe qu'il n'y paraît. La fonction sigmoid pour modéliser la probabilité de réussite m'a bien aidé.

---

## 🔮 Améliorations prévues

- [ ] 🧠 **Deep Learning** — remplacer le RF par un LSTM pour tenir compte de la séquence temporelle
- [ ] 🗄️ **Base de données** — SQLite ou PostgreSQL pour persister les profils (là tout est en RAM)
- [ ] 🌐 **Interface web** — front React ou Vue.js
- [ ] ❓ **Vraies questions** — base de 500+ questions réelles par sujet
- [ ] 🔐 **Authentification** — JWT ou OAuth2
- [ ] 🧪 **Tests unitaires** — pytest (j'ai pas trop eu le temps...)

---

## 📚 Stack technique

| Composant | Technologie |
|-----------|-------------|
| API REST | FastAPI + Uvicorn |
| ML | Scikit-learn (RandomForest) |
| Data | Pandas + NumPy |
| Validation | Pydantic v2 |
| Analyse | Jupyter + Matplotlib + Seaborn |

---

<div align="center">

*Projet personnel — ESIEA 3A, 2025-2026*  
*Fait avec ☕ et beaucoup de Stack Overflow*

</div>
