"""
retriever.py
Recherche des passages pertinents dans ChromaDB
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

from src.embeddings import get_embedding_model, get_chroma_client

load_dotenv()


def retrieve(
    query: str,
    collection_name: str,
    n_results: int = 3
) -> list[dict]:
    """
    Recherche les passages les plus pertinents pour une question.

    Args:
        query: La question posée par l'utilisateur
        collection_name: La collection ChromaDB où chercher
        n_results: Nombre de passages à retourner

    Returns:
        Liste de dicts avec 'text', 'score', 'metadata'
    """
    model = get_embedding_model()
    client = get_chroma_client()

    # Vérifie que la collection existe
    existing = [col.name for col in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(f"Collection introuvable : '{collection_name}'. "
                         f"Collections disponibles : {existing}")

    collection = client.get_collection(name=collection_name)

    # Vectorise la question
    query_vector = model.encode(query).tolist()

    # Recherche dans ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, collection.count())
    )

    # Formate les résultats
    passages = []
    for i in range(len(results["documents"][0])):
        passages.append({
            "text": results["documents"][0][i],
            "score": round(1 - results["distances"][0][i], 4),  # score de similarité
            "metadata": results["metadatas"][0][i]
        })

    print(f"🔍 {len(passages)} passages trouvés pour : '{query}'")
    for i, p in enumerate(passages):
        print(f"  [{i+1}] Score: {p['score']} | {p['text'][:80]}...")

    return passages


def format_context(passages: list[dict]) -> str:
    """
    Formate les passages récupérés en un contexte lisible pour le LLM.

    Args:
        passages: Liste de passages (depuis retrieve())

    Returns:
        Texte formaté à injecter dans le prompt
    """
    context = ""
    for i, p in enumerate(passages):
        context += f"[Extrait {i+1}]\n{p['text']}\n\n"
    return context.strip()


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    from src.ingestion import load_and_split
    from src.embeddings import embed_and_store

    # Données de test
    chunks = [
        "Développeur Python avec 5 ans d'expérience en machine learning.",
        "Compétences : LangChain, Docker, AWS, FastAPI, ChromaDB.",
        "Expérience en déploiement de modèles en production avec MLflow.",
        "Formation Master MIAGE à l'Université de Bordeaux.",
        "Recherche un poste de ML Engineer dans le domaine de la santé.",
    ]

    embed_and_store(chunks, collection_name="test_retriever")

    query = "Quelles sont les compétences techniques du candidat ?"
    passages = retrieve(query, collection_name="test_retriever")

    print("\n--- Contexte formaté ---")
    print(format_context(passages))