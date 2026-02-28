"""
Tests unitaires pour embeddings.py
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import get_embedding_model, embed_and_store, list_collections


def test_embedding_model_loads():
    """Vérifie que le modèle se charge correctement"""
    model = get_embedding_model()
    assert model is not None


def test_embed_single_chunk():
    """Vérifie qu'un texte est bien vectorisé"""
    model = get_embedding_model()
    vector = model.encode("Développeur Python senior")
    assert len(vector) > 0
    assert hasattr(vector[0], '__float__')  # fonctionne pour numpy float aussi


def test_embed_and_store():
    """Vérifie le stockage complet dans ChromaDB"""
    chunks = [
        "Compétences Python et machine learning.",
        "Expérience Docker et déploiement cloud.",
        "Formation Master en informatique."
    ]
    collection = embed_and_store(
        chunks=chunks,
        collection_name="test_unit",
        metadata={"type": "test"}
    )
    assert collection.count() == 3


def test_list_collections():
    """Vérifie que les collections sont listables"""
    collections = list_collections()
    assert isinstance(collections, list)