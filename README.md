---
title: Fair Hire
emoji: ⚖️
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: false
---
# ⚖️ Fair Hire — Assistant RH Inclusif basé sur l'IA

> Détection de biais dans les offres d'emploi et matching intelligent CV/Poste  
> Powered by RAG · LangChain · Mistral · ChromaDB · MLflow

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![MLflow](https://img.shields.io/badge/MLflow-3.10-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)

---

## 🎯 Problème résolu

Les offres d'emploi contiennent souvent des biais inconscients :
- **Langage genré** : "ninja", "rockstar", "ambitieux"
- **Critères discriminants** : tranches d'âge, "Grande École obligatoire"
- **Formulations excluantes** : "disponible immédiatement", "obligatoire"

En France, ces biais réduisent la diversité des candidatures et exposent les entreprises à des risques juridiques (loi sur la non-discrimination à l'embauche).

**Fair Hire** analyse automatiquement les offres d'emploi, détecte ces biais et génère des suggestions de reformulation inclusives. Il effectue également un matching intelligent entre un CV et une offre.

---

## 🏗️ Architecture
```
PDF (CV / Offre)
      ↓
Ingestion (PyMuPDF)
      ↓
Chunking (LangChain RecursiveCharacterTextSplitter)
      ↓
Embeddings (sentence-transformers: all-MiniLM-L6-v2)
      ↓
Stockage vectoriel (ChromaDB)
      ↓
Retrieval sémantique
      ↓
Génération (Mistral 7B via Ollama)
      ↓
Rapport structuré (Biais + Matching)
      ↓
Tracking (MLflow)
```

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|------------|
| LLM local | Mistral 7B via Ollama |
| Orchestration RAG | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Base vectorielle | ChromaDB |
| Lecture PDF | PyMuPDF |
| Interface | Streamlit |
| Tracking MLOps | MLflow |
| Conteneurisation | Docker + docker-compose |
| CI/CD | GitHub Actions + Jenkins |
| Tests | pytest |

---

## 📁 Structure du projet
```
fair-hire/
├── src/
│   ├── ingestion.py        # Lecture et découpage des PDFs
│   ├── embeddings.py       # Vectorisation et stockage ChromaDB
│   ├── retriever.py        # Recherche sémantique
│   ├── generator.py        # Génération via Mistral
│   ├── bias_detector.py    # Détection de biais
│   ├── agent.py            # Orchestration du pipeline
│   └── mlflow_tracker.py   # Tracking des expériences
├── app/
│   └── streamlit_app.py    # Interface utilisateur
├── tests/                  # Tests unitaires (pytest)
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
└── .github/workflows/ci.yml
```

---

## 🚀 Installation et lancement

### Prérequis
- Python 3.12+
- Docker
- [Ollama](https://ollama.com) avec Mistral installé

### 1. Clone le repo
```bash
git clone https://github.com/bibakali/fair-hire.git
cd fair-hire
```

### 2. Installe les dépendances
```bash
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure les variables d'environnement
```bash
cp .env.example .env
```

### 4. Lance Ollama
```bash
ollama pull mistral
ollama serve
```

### 5. Lance l'application
```bash
PYTHONPATH=/chemin/vers/fair-hire streamlit run app/streamlit_app.py
```

Ouvre **http://localhost:8501**

### 6. Lance avec Docker
```bash
docker-compose up --build
```

---

## 📊 MLflow — Tracking des expériences
```bash
mlflow ui --backend-store-uri sqlite:////chemin/vers/fair-hire/mlflow.db --port 5000
```

Ouvre **http://127.0.0.1:5000**

Métriques trackées :
- `bias_score` — score de biais de l'offre (0 = neutre)
- `n_gendered_words` — nombre de mots genrés détectés
- `n_discriminatory_patterns` — patterns discriminatoires
- `duration_seconds` — durée du pipeline

---

## 🧪 Tests
```bash
pytest tests/ -v
```

Couverture des tests :
- `test_ingestion.py` — lecture et découpage PDFs
- `test_embeddings.py` — vectorisation ChromaDB
- `test_retriever.py` — recherche sémantique
- `test_generator.py` — construction des prompts
- `test_bias_detector.py` — détection de biais
- `test_agent.py` — orchestration pipeline

---



## 💡 Choix techniques et compromis

**Pourquoi Mistral 7B local plutôt qu'OpenAI ?**
Confidentialité des données RH — les CVs et offres d'emploi ne quittent pas la machine.

**Pourquoi ChromaDB plutôt que FAISS ?**
ChromaDB offre une persistance native et une API plus simple pour un projet de cette taille. FAISS serait plus performant à grande échelle.

**Pourquoi chunk_size=512 ?**
Après tests comparatifs, 512 offre le meilleur équilibre entre contexte suffisant et précision du retrieval sur des documents RH.

**Pourquoi all-MiniLM-L6-v2 ?**
Modèle léger (90Mo), rapide, et performant sur du texte professionnel en anglais et français.

---

## 🔮 Améliorations futures

- Évaluation de la qualité RAG avec **RAGAs**
- Support multilingue amélioré
- Fine-tuning du détecteur de biais sur corpus RH français
- API REST avec FastAPI
- Déploiement Hugging Face Spaces

---

**🤖 Optimiseur ATS**
    Détecte les mots-clés manquants et réécrit ton CV pour passer les filtres automatiques.

## 👩‍💻 Auteure

**Habibatou BA** — ML Engineer | MLOps & GenAI  
[LinkedIn](https://linkedin.com/in/habibatou-ba) · [GitHub](https://github.com/bibakali)