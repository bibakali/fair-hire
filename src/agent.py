"""
agent.py
Orchestration du pipeline RAG complet — cerveau du projet Fair Hire
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from src.ingestion import load_and_split
from src.embeddings import embed_and_store
from src.retriever import retrieve, format_context
from src.generator import generate, generate_matching_report
from src.bias_detector import analyze, format_report

load_dotenv()


# ---------------------------------------------------------------
# Dataclass pour le résultat final
# ---------------------------------------------------------------

@dataclass
class FairHireResult:
    cv_filename: str = ""
    job_filename: str = ""
    bias_report: str = ""
    bias_score: float = 0.0
    matching_report: str = ""
    cv_summary: str = ""
    job_summary: str = ""
    status: str = "pending"
    error: str = ""


# ---------------------------------------------------------------
# Outils individuels (appelables séparément)
# ---------------------------------------------------------------

def tool_load_document(file_path: str, doc_type: str) -> list[str]:
    """
    Outil 1 : Charge et découpe un document PDF.

    Args:
        file_path: Chemin vers le PDF
        doc_type: 'cv' ou 'job'

    Returns:
        Liste de chunks
    """
    print(f"\n🔧 [Outil 1] Chargement du {doc_type} : {file_path}")
    chunks = load_and_split(file_path)
    return chunks


def tool_vectorize(chunks: list[str], collection_name: str, metadata: dict) -> None:
    """
    Outil 2 : Vectorise et stocke les chunks dans ChromaDB.

    Args:
        chunks: Liste de morceaux de texte
        collection_name: Nom de la collection
        metadata: Métadonnées du document
    """
    print(f"\n🔧 [Outil 2] Vectorisation → collection '{collection_name}'")
    embed_and_store(chunks, collection_name=collection_name, metadata=metadata)


def tool_retrieve_context(query: str, collection_name: str, n_results: int = 3) -> str:
    """
    Outil 3 : Récupère les passages pertinents pour une question.

    Args:
        query: Question ou critère de recherche
        collection_name: Collection où chercher
        n_results: Nombre de passages

    Returns:
        Contexte formaté
    """
    print(f"\n🔧 [Outil 3] Recherche : '{query}' dans '{collection_name}'")
    passages = retrieve(query, collection_name, n_results)
    return format_context(passages)


def tool_detect_bias(text: str) -> tuple[str, float]:
    """
    Outil 4 : Détecte les biais dans une offre d'emploi.

    Args:
        text: Texte de l'offre

    Returns:
        Tuple (rapport formaté, score de biais)
    """
    print(f"\n🔧 [Outil 4] Détection des biais...")
    report = analyze(text)
    return format_report(report), report.bias_score


def tool_generate_summary(context: str, doc_type: str) -> str:
    """
    Outil 5 : Génère un résumé structuré d'un document.

    Args:
        context: Contexte extrait
        doc_type: 'cv' ou 'job'

    Returns:
        Résumé généré par Mistral
    """
    print(f"\n🔧 [Outil 5] Génération du résumé {doc_type}...")
    if doc_type == "cv":
        question = "Résume ce CV : compétences clés, expériences, formation, points forts."
    else:
        question = "Résume cette offre d'emploi : poste, compétences requises, contexte."

    return generate(question, context, mode="general")


# ---------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------

def run_pipeline(cv_path: str, job_path: str) -> FairHireResult:
    """
    Pipeline complet Fair Hire :
    1. Charge les documents
    2. Vectorise et stocke
    3. Détecte les biais
    4. Génère les résumés
    5. Génère le rapport de matching

    Args:
        cv_path: Chemin vers le CV (PDF)
        job_path: Chemin vers l'offre d'emploi (PDF)

    Returns:
        FairHireResult avec tous les résultats
    """
    result = FairHireResult(
        cv_filename=os.path.basename(cv_path),
        job_filename=os.path.basename(job_path)
    )

    try:
        # --- Étape 1 : Chargement des documents ---
        print("\n" + "="*50)
        print("ÉTAPE 1 : Chargement des documents")
        print("="*50)
        cv_chunks = tool_load_document(cv_path, "cv")
        job_chunks = tool_load_document(job_path, "job")

        # --- Étape 2 : Vectorisation ---
        print("\n" + "="*50)
        print("ÉTAPE 2 : Vectorisation")
        print("="*50)
        tool_vectorize(cv_chunks, "cv_current", {"type": "cv", "file": cv_path})
        tool_vectorize(job_chunks, "job_current", {"type": "job", "file": job_path})

        # --- Étape 3 : Détection des biais ---
        print("\n" + "="*50)
        print("ÉTAPE 3 : Détection des biais")
        print("="*50)
        job_text = " ".join(job_chunks)
        result.bias_report, result.bias_score = tool_detect_bias(job_text)

        # --- Étape 4 : Résumés ---
        print("\n" + "="*50)
        print("ÉTAPE 4 : Génération des résumés")
        print("="*50)
        cv_context = tool_retrieve_context(
            "compétences expériences formation", "cv_current"
        )
        job_context = tool_retrieve_context(
            "compétences requises poste missions", "job_current"
        )
        result.cv_summary = tool_generate_summary(cv_context, "cv")
        result.job_summary = tool_generate_summary(job_context, "job")

        # --- Étape 5 : Matching ---
        print("\n" + "="*50)
        print("ÉTAPE 5 : Rapport de matching")
        print("="*50)
        result.matching_report = generate_matching_report(cv_context, job_context)

        result.status = "success"
        print("\n✅ Pipeline terminé avec succès !")

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        print(f"\n❌ Erreur pipeline : {e}")

    return result


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/agent.py <cv.pdf> <offre.pdf>")
        sys.exit(1)

    result = run_pipeline(sys.argv[1], sys.argv[2])
    print("\n--- RÉSULTAT FINAL ---")
    print(f"Status : {result.status}")
    print(f"Bias score : {result.bias_score}")
    print(f"\n{result.bias_report}")
    print(f"\n{result.matching_report}")