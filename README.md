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

<div align="center">

# ⚖️ Fair Hire — Assistant RH Inclusif

> Détection de biais · Matching CV/Offre · Optimisation ATS · Propulsé par RAG + Mistral

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/bibakali/fair-hire)
[![GitHub](https://img.shields.io/badge/GitHub-bibakali%2Ffair--hire-black?logo=github)](https://github.com/bibakali/fair-hire)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Tests](https://img.shields.io/badge/Tests-27%20passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

</div>

## 🎯 Problème résolu

En France, **les offres d'emploi contiennent fréquemment des biais inconscients** qui réduisent la diversité des candidatures et exposent les entreprises à des risques juridiques :

- **Langage genré** : "ninja", "rockstar", "ambitieux", "combatif"
- **Critères discriminants** : tranches d'âge, "Grande École obligatoire"
- **Formulations excluantes** : "disponible immédiatement", "obligatoire"

Et côté candidats, **70% des CVs sont filtrés par des logiciels ATS** avant même d'être lus par un humain — souvent à cause de mots-clés manquants.

**Fair Hire** adresse les deux problèmes en même temps.

---

## ✨ Fonctionnalités

### 🔍 Analyse de biais
Détecte automatiquement les formulations discriminantes dans une offre d'emploi.
- Score de biais de 0 à 1
- Mots genrés et patterns discriminatoires identifiés
- Suggestions de reformulation inclusive
- Résultat instantané

### 🎯 Matching CV/Offre
Rapport de matching rapide entre un CV et une offre d'emploi.
- Score de correspondance sur 10
- Points forts du candidat pour ce poste
- Points à développer
- Recommandation finale

### 📊 Pipeline complet
Analyse approfondie combinant matching + biais + résumés.
- Toutes les analyses en une fois
- Dashboard avec métriques clés
- Résumés structurés du CV et de l'offre

### 🤖 Optimiseur ATS
La feature la plus différenciante — optimise le CV pour passer les filtres automatiques.
- Extraction des mots-clés de l'offre
- Identification des mots-clés manquants dans le CV
- Réécriture intelligente des expériences par Mistral
- Score ATS avant/après

---

## 🏗️ Architecture RAG + Agent
```
PDF (CV / Offre) ou Texte collé
          ↓
   Ingestion (PyMuPDF)
          ↓
   Chunking (LangChain)
          ↓
   Embeddings (sentence-transformers)
          ↓
   Stockage vectoriel (ChromaDB)
          ↓
   Retrieval sémantique
          ↓
   Agent LangChain — orchestre les outils
          ↓
   Génération (Mistral via API)
          ↓
   Rapport structuré
          ↓
   Tracking (MLflow)
```

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|------------|
| LLM | Mistral via API Mistral AI |
| Orchestration RAG | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Base vectorielle | ChromaDB |
| Lecture PDF | PyMuPDF |
| Interface | Streamlit |
| Tracking MLOps | MLflow |
| Conteneurisation | Docker + docker-compose |
| CI/CD | GitHub Actions + Jenkins |
| Tests | pytest (27 tests) |
| Déploiement | Hugging Face Spaces |

---

## 🚀 Lancement local

### Prérequis
- Python 3.12+
- Docker
- Clé API [Mistral AI](https://console.mistral.ai)

### Installation
```bash
git clone https://github.com/bibakali/fair-hire.git
cd fair-hire
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Edite .env et ajoute ta clé MISTRAL_API_KEY
```

### Lancement
```bash
./run.sh
```

Ouvre **http://localhost:8501**

### Avec Docker
```bash
docker-compose up --build
```

---

## 🧪 Tests
```bash
pytest tests/ -v
```

27 tests couvrant tous les modules :
`ingestion` · `embeddings` · `retriever` · `generator` · `bias_detector` · `agent`

---

## 📊 MLflow — Tracking des expériences
```bash
mlflow ui --backend-store-uri sqlite:////chemin/fair-hire/mlflow.db --port 5000
```

Métriques trackées : `bias_score` · `n_gendered_words` · `duration_seconds` · `ats_score`

---

## 💡 Décisions techniques

**Pourquoi Mistral plutôt qu'OpenAI ?**
Mistral est une entreprise française — cohérent avec la problématique RH France. API moins chère et performances comparables sur du texte professionnel français.

**Pourquoi ChromaDB plutôt que FAISS ?**
Persistance native et API simple pour ce volume de données. FAISS serait plus performant à grande échelle.

**Pourquoi chunk_size=512 ?**
Meilleur équilibre contexte/précision sur des documents RH après tests comparatifs loggués dans MLflow.

**Pourquoi all-MiniLM-L6-v2 ?**
Léger (90Mo), rapide, et performant sur du texte professionnel français et anglais.

---

## 🔮 Roadmap

- [ ] Téléchargement du CV optimisé (PDF + DOCX)
- [ ] Évaluation RAG avec RAGAs
- [ ] Génération de lettre de motivation
- [ ] Préparation aux questions d'entretien
- [ ] Support multilingue

---

## 👩‍💻 Auteure

**Habibatou BA** — ML Engineer | MLOps & GenAI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-habibatou--ba-blue?logo=linkedin)](https://linkedin.com/in/habibatou-ba)
[![GitHub](https://img.shields.io/badge/GitHub-bibakali-black?logo=github)](https://github.com/bibakali)