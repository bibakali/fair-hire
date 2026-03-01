"""
generator.py
Génération de réponses via Mistral API (cloud) ou Ollama (local)
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
USE_API = os.getenv("USE_MISTRAL_API", "false").lower() == "true"


def build_prompt(question: str, context: str, mode: str = "general") -> str:
    if mode == "matching":
        system = """Tu es un expert RH spécialisé dans l'analyse de CVs et d'offres d'emploi.
Réponds uniquement à partir des extraits fournis. Sois précis et structuré."""
    elif mode == "bias":
        system = """Tu es un expert en diversité et inclusion dans le recrutement.
Détecte les biais potentiels et propose des alternatives inclusives."""
    else:
        system = """Tu es un assistant RH expert.
Réponds uniquement à partir des extraits fournis."""

    prompt = f"""<s>[INST] {system}

### Extraits du document :
{context}

### Question :
{question}

Réponds de manière structurée et précise. [/INST]"""

    return prompt


def call_mistral_api(prompt: str, max_tokens: int = 500) -> str:
    """Appel à l'API Mistral cloud."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    response = requests.post(
        MISTRAL_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def call_ollama(prompt: str) -> str:
    """Appel à Ollama en local avec streaming."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        },
        stream=True,
        timeout=600
    )
    response.raise_for_status()

    result = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            result += data.get("response", "")
            if data.get("done", False):
                break
    return result.strip()


def generate(question: str, context: str, mode: str = "general") -> str:
    """Génère une réponse via API Mistral ou Ollama selon la config."""
    prompt = build_prompt(question, context, mode)

    try:
        if USE_API:
            return call_mistral_api(prompt)
        else:
            return call_ollama(prompt)

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Impossible de contacter le LLM.")
    except requests.exceptions.Timeout:
        raise TimeoutError("Mistral met trop de temps à répondre. Réessaie.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération : {e}")


def generate_matching_report(cv_context: str, job_context: str) -> str:
    """Génère un rapport de matching CV / Offre d'emploi."""
    full_prompt = f"""Tu es un expert RH français. Analyse en français la correspondance entre ce candidat et ce poste.

### CV :
{cv_context[:1000]}

### OFFRE :
{job_context[:1000]}

Réponds UNIQUEMENT avec ce format markdown :

## Score : X/10

## ✅ Points forts
- point 1
- point 2
- point 3

## ⚠️ Points à développer
- point 1
- point 2

## 📋 Recommandation
Une phrase de conclusion."""

    try:
        if USE_API:
            return call_mistral_api(full_prompt, max_tokens=600)
        else:
            return call_ollama(full_prompt)

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Impossible de contacter le LLM.")
    except requests.exceptions.Timeout:
        raise TimeoutError("Mistral met trop de temps à répondre. Réessaie.")