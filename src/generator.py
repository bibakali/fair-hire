"""
generator.py
Génération de réponses via Mistral (Ollama)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def build_prompt(question: str, context: str, mode: str = "general") -> str:
    """
    Construit le prompt envoyé à Mistral selon le mode.

    Args:
        question: La question de l'utilisateur
        context: Les passages récupérés par le retriever
        mode: 'general', 'matching', ou 'bias'

    Returns:
        Prompt formaté
    """
    if mode == "matching":
        system = """Tu es un expert RH spécialisé dans l'analyse de CVs et d'offres d'emploi.
Tu dois analyser objectivement la correspondance entre un candidat et un poste.
Réponds uniquement à partir des extraits fournis.
Sois précis, structuré et cite les éléments du document qui justifient ton analyse."""

    elif mode == "bias":
        system = """Tu es un expert en diversité et inclusion dans le recrutement.
Tu dois détecter les biais potentiels dans les offres d'emploi :
- Langage genré (ex: "ninja", "rockstar", adjectifs masculins)
- Critères non pertinents (âge, apparence)
- Formulations excluantes
Sois factuel et propose des alternatives inclusives."""

    else:
        system = """Tu es un assistant RH expert.
Réponds uniquement à partir des extraits fournis.
Si l'information n'est pas dans les extraits, dis-le clairement."""

    prompt = f"""<s>[INST] {system}

### Extraits du document :
{context}

### Question :
{question}

Réponds de manière structurée et précise. [/INST]"""

    return prompt


def generate(question: str, context: str, mode: str = "general") -> str:
    """
    Envoie le prompt à Mistral et retourne la réponse.

    Args:
        question: La question de l'utilisateur
        context: Le contexte récupéré par le retriever
        mode: 'general', 'matching', ou 'bias'

    Returns:
        Réponse générée par Mistral
    """
    prompt = build_prompt(question, context, mode)

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Réponses précises et reproductibles
                    "top_p": 0.9,
                }
            },
            timeout=600
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Impossible de contacter Ollama sur {OLLAMA_HOST}. "
            f"Lance Ollama avec : ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Mistral met trop de temps à répondre. Réessaie.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération : {e}")


def generate_matching_report(cv_context: str, job_context: str) -> str:
        """
        Génère un rapport de matching avec streaming pour éviter le timeout.
        """
        full_prompt = f"""<s>[INST] Tu es un expert RH français. Analyse en français la correspondance entre ce candidat et ce poste.
    Sois concis et structuré.

    ### CV :
    {cv_context[:1000]}

    ### OFFRE :
    {job_context[:1000]}

    Réponds UNIQUEMENT avec ce format :

    ## Score : X/10

    ## Points forts
    - point 1
    - point 2
    - point 3

    ## Points à développer
    - point 1
    - point 2

    ## Recommandation
    Une phrase. [/INST]"""

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": "mistral",
                    "prompt": full_prompt,
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
                    import json
                    data = json.loads(line)
                    result += data.get("response", "")
                    if data.get("done", False):
                        break

            return result.strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Impossible de contacter Ollama. Lance : ollama serve")
        except requests.exceptions.Timeout:
            raise TimeoutError("Mistral met trop de temps à répondre. Réessaie.")

# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    context = """[Extrait 1]
Développeur Python avec 5 ans d'expérience en machine learning et MLOps.

[Extrait 2]
Compétences : LangChain, Docker, AWS, FastAPI, ChromaDB, MLflow."""

    question = "Quelles sont les compétences techniques de ce candidat ?"
    reponse = generate(question, context, mode="general")
    print(reponse)