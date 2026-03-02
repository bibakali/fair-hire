"""
mlflow_tracker.py
Tracking des expériences avec MLflow
"""

import os
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow non disponible — tracking désactivé")

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fair-hire-rag"




def setup_mlflow():
    if not MLFLOW_AVAILABLE:
        return
    """Configure MLflow en mode local (pas de serveur requis)."""
    mlflow.set_tracking_uri("sqlite:////Users/habiba/fair-hire/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow configuré en mode local : ./mlruns")

def log_ingestion_run(
    file_name: str,
    doc_type: str,
    chunk_size: int,
    chunk_overlap: int,
    n_chunks: int,
    n_pages: int
):
    """
    Logge un run d'ingestion de document.

    Args:
        file_name: Nom du fichier
        doc_type: 'cv' ou 'job'
        chunk_size: Taille des chunks
        chunk_overlap: Chevauchement
        n_chunks: Nombre de chunks générés
        n_pages: Nombre de pages du document
    """
    if not MLFLOW_AVAILABLE:
        return
    setup_mlflow()

    with mlflow.start_run(run_name=f"ingestion_{doc_type}_{datetime.now().strftime('%H%M%S')}"):
        # Paramètres
        mlflow.log_param("file_name", file_name)
        mlflow.log_param("doc_type", doc_type)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)

        # Métriques
        mlflow.log_metric("n_chunks", n_chunks)
        mlflow.log_metric("n_pages", n_pages)
        mlflow.log_metric("chunks_per_page", round(n_chunks / max(n_pages, 1), 2))

        print(f"📊 MLflow — ingestion loggée : {n_chunks} chunks, {n_pages} pages")


def log_retrieval_run(
    query: str,
    collection_name: str,
    n_results: int,
    top_score: float,
    avg_score: float
):
    """
    Logge un run de retrieval.

    Args:
        query: La question posée
        collection_name: Collection interrogée
        n_results: Nombre de résultats retournés
        top_score: Meilleur score de similarité
        avg_score: Score moyen
    """
    if not MLFLOW_AVAILABLE:
        return
    setup_mlflow()

    with mlflow.start_run(run_name=f"retrieval_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param("query", query[:100])  # tronque si trop long
        mlflow.log_param("collection", collection_name)
        mlflow.log_param("n_results", n_results)

        mlflow.log_metric("top_score", top_score)
        mlflow.log_metric("avg_score", avg_score)

        print(f"📊 MLflow — retrieval loggué : top_score={top_score}, avg={avg_score}")


def log_bias_run(
    file_name: str,
    bias_score: float,
    n_gendered_words: int,
    n_discriminatory_patterns: int
):
    """
    Logge un run d'analyse de biais.

    Args:
        file_name: Nom du fichier analysé
        bias_score: Score de biais calculé
        n_gendered_words: Nombre de mots genrés
        n_discriminatory_patterns: Nombre de patterns discriminatoires
    """
    if not MLFLOW_AVAILABLE:
        return
    setup_mlflow()

    with mlflow.start_run(run_name=f"bias_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param("file_name", file_name)

        mlflow.log_metric("bias_score", bias_score)
        mlflow.log_metric("n_gendered_words", n_gendered_words)
        mlflow.log_metric("n_discriminatory_patterns", n_discriminatory_patterns)

        # Tag automatique selon le niveau de biais
        if bias_score == 0:
            mlflow.set_tag("bias_level", "none")
        elif bias_score < 0.05:
            mlflow.set_tag("bias_level", "low")
        else:
            mlflow.set_tag("bias_level", "high")

        print(f"📊 MLflow — biais loggué : score={bias_score}, level={mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get('bias_level')}")


def log_pipeline_run(
    cv_file: str,
    job_file: str,
    bias_score: float,
    pipeline_status: str,
    duration_seconds: float,
    chunk_size: int = 512
):
    """
    Logge un run complet du pipeline Fair Hire.

    Args:
        cv_file: Nom du CV
        job_file: Nom de l'offre
        bias_score: Score de biais de l'offre
        pipeline_status: 'success' ou 'error'
        duration_seconds: Durée du pipeline
        chunk_size: Taille des chunks utilisée
    """
    if not MLFLOW_AVAILABLE:
        return
    setup_mlflow()

    with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%H%M%S')}"):
        # Paramètres
        mlflow.log_param("cv_file", cv_file)
        mlflow.log_param("job_file", job_file)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("status", pipeline_status)

        # Métriques
        mlflow.log_metric("bias_score", bias_score)
        mlflow.log_metric("duration_seconds", duration_seconds)

        # Tags
        mlflow.set_tag("pipeline_version", "1.0")
        mlflow.set_tag("model", "mistral")
        mlflow.set_tag("embedding_model", "all-MiniLM-L6-v2")

        print(f"📊 MLflow — pipeline loggué : status={pipeline_status}, duration={duration_seconds}s")


# Test rapide
if __name__ == "__main__":
    print("Test MLflow tracking...")
    log_bias_run(
        file_name="test_offre.pdf",
        bias_score=0.042,
        n_gendered_words=2,
        n_discriminatory_patterns=1
    )
    print("✅ Run loggué — lance 'mlflow ui' pour voir le dashboard")