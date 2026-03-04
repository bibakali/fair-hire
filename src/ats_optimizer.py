"""
ats_optimizer.py
Extracteur de mots-clés ATS et réécriture du CV pour matcher l'offre
"""

from dataclasses import dataclass, field

try:
    from src.generator import call_mistral_api, call_ollama, USE_API
except ImportError:
    from generator import call_mistral_api, call_ollama, USE_API

# ---------------------------------------------------------------
# Mots-clés techniques courants en Data/ML
# ---------------------------------------------------------------

TECH_KEYWORDS = [
    "python", "sql", "r", "scala", "java",
    "machine learning", "deep learning", "nlp", "computer vision",
    "scikit-learn", "pytorch", "tensorflow", "keras", "xgboost",
    "lightgbm", "catboost", "hugging face", "transformers",
    "mlflow", "kubeflow", "airflow", "docker", "kubernetes",
    "jenkins", "ci/cd", "github actions", "fastapi", "flask",
    "aws", "gcp", "azure", "s3", "ec2", "sagemaker",
    "spark", "hadoop", "kafka", "elasticsearch", "dbt",
    "snowflake", "bigquery", "postgresql", "mongodb",
    "llm", "rag", "langchain", "openai", "mistral", "llama",
    "fine-tuning", "prompt engineering", "chromadb", "embeddings",
    "tableau", "power bi", "streamlit", "dash",
    "agile", "scrum", "git", "api", "rest", "microservices",
]

# ---------------------------------------------------------------
# Dataclass résultat
# ---------------------------------------------------------------

@dataclass
class ATSReport:
    keywords_in_offer: list[str] = field(default_factory=list)
    keywords_in_cv: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    ats_score: float = 0.0
    summary: str = ""


# ---------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------

def extract_keywords(text: str) -> list[str]:
    """Extrait les mots-clés techniques d'un texte."""
    text_lower = text.lower()
    return list({kw for kw in TECH_KEYWORDS if kw in text_lower})


def compute_ats_score(cv_keywords: list[str], offer_keywords: list[str]) -> float:
    if not offer_keywords:
        return 1.0
    matches = set(cv_keywords) & set(offer_keywords)
    return round(len(matches) / len(offer_keywords), 2)


def analyze_ats(cv_text: str, job_text: str) -> ATSReport:
    report = ATSReport()
    report.keywords_in_offer = extract_keywords(job_text)
    report.keywords_in_cv = extract_keywords(cv_text)
    report.missing_keywords = [
        kw for kw in report.keywords_in_offer
        if kw not in report.keywords_in_cv
    ]
    report.ats_score = compute_ats_score(
        report.keywords_in_cv,
        report.keywords_in_offer
    )
    if report.ats_score >= 0.8:
        report.summary = "✅ Excellent score ATS — ton CV est bien aligné avec l'offre."
    elif report.ats_score >= 0.5:
        report.summary = f"⚠️ Score ATS moyen — {len(report.missing_keywords)} mots-clés manquants."
    else:
        report.summary = f"🚨 Score ATS faible — {len(report.missing_keywords)} mots-clés manquants."
    return report


def rewrite_cv_for_ats(cv_text: str, missing_keywords: list[str], job_text: str) -> str:
    if not missing_keywords:
        return "✅ Ton CV contient déjà tous les mots-clés importants de l'offre !"

    keywords_str = ", ".join(missing_keywords[:10])

    prompt = f"""Tu es un expert en rédaction de CV pour des postes Data/ML.

Le candidat postule à cette offre et son CV manque ces mots-clés : {keywords_str}

CV actuel :
{cv_text[:800]}

Offre d'emploi :
{job_text[:500]}

Réécris 3 bullet points du CV en intégrant NATURELLEMENT ces mots-clés.
Ne mens pas — utilise uniquement les vraies expériences du candidat.
Reformule pour mettre en valeur les compétences existantes avec le bon vocabulaire.

Format :

## ✏️ Bullet points réécrits

**Avant :**
- bullet point original

**Après :**
- bullet point réécrit avec les mots-clés

## 💡 Conseil
Une phrase de conseil personnalisé."""

    try:
        if USE_API:
            return call_mistral_api(prompt, max_tokens=600)
        else:
            return call_ollama(prompt)
    except Exception as e:
        return f"Erreur lors de la réécriture : {e}"