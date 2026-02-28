"""
embeddings.py
Transformation du texte en vecteurs numériques et stockage dans ChromaDB
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()

# Modèle d'embedding léger et efficace
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


def get_embedding_model() -> SentenceTransformer:
    """
    Charge le modèle d'embedding.
    Téléchargé automatiquement au premier appel (~90Mo).
    """
    print(f"📦 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def get_chroma_client() -> chromadb.Client:
    """
    Initialise le client ChromaDB en mode persistant.
    Les vecteurs sont sauvegardés sur disque dans CHROMA_PATH.
    """
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def embed_and_store(
    chunks: list[str],
    collection_name: str,
    metadata: dict = None
) -> chromadb.Collection:
    """
    Vectorise les morceaux de texte et les stocke dans ChromaDB.

    Args:
        chunks: Liste de morceaux de texte (depuis ingestion.py)
        collection_name: Nom de la collection ChromaDB (ex: "cv_john", "offre_dev")
        metadata: Infos supplémentaires sur le document (ex: type, nom fichier)

    Returns:
        La collection ChromaDB créée
    """
    model = get_embedding_model()
    client = get_chroma_client()

    # Supprime la collection si elle existe déjà (rechargement propre)
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️  Collection existante supprimée : {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    # Génération des embeddings
    print(f"⚙️  Vectorisation de {len(chunks)} morceaux...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    # Préparation des métadonnées
    meta = metadata or {}
    metadatas = [{**meta, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]

    # Stockage dans ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ {len(chunks)} vecteurs stockés dans la collection '{collection_name}'")
    return collection


def list_collections() -> list[str]:
    """Retourne la liste des collections disponibles dans ChromaDB."""
    client = get_chroma_client()
    collections = client.list_collections()
    return [col.name for col in collections]


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    test_chunks = [
        "Développeur Python avec 5 ans d'expérience en machine learning.",
        "Compétences : LangChain, Docker, AWS, FastAPI.",
        "Expérience en déploiement de modèles en production.",
    ]

    collection = embed_and_store(
        chunks=test_chunks,
        collection_name="test_collection",
        metadata={"type": "cv", "filename": "test"}
    )

    print(f"\n📋 Collections disponibles : {list_collections()}")
    print(f"📊 Nombre de vecteurs : {collection.count()}")