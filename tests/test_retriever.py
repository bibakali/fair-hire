"""
Tests unitaires pour retriever.py
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import embed_and_store
from src.retriever import retrieve, format_context


@pytest.fixture(scope="module")
def setup_collection():
    """Crée une collection de test réutilisable pour tous les tests"""
    chunks = [
        "Développeur Python avec 5 ans d'expérience.",
        "Compétences Docker, AWS et déploiement cloud.",
        "Formation Master informatique Bordeaux.",
        "Expérience en machine learning et MLflow.",
        "Recherche poste ML Engineer Paris.",
    ]
    embed_and_store(chunks, collection_name="test_retriever_unit")
    return "test_retriever_unit"


def test_retrieve_returns_results(setup_collection):
    """Vérifie que la recherche retourne des résultats"""
    results = retrieve("compétences Python", setup_collection)
    assert len(results) > 0


def test_retrieve_has_correct_keys(setup_collection):
    """Vérifie la structure des résultats"""
    results = retrieve("expérience cloud", setup_collection)
    for r in results:
        assert "text" in r
        assert "score" in r
        assert "metadata" in r

def test_retrieve_score_is_float(setup_collection):
    """Vérifie que les scores sont des nombres"""
    results = retrieve("machine learning", setup_collection)
    for r in results:
        assert isinstance(r["score"], float)


def test_format_context(setup_collection):
    """Vérifie que le contexte est bien formaté"""
    results = retrieve("formation", setup_collection)
    context = format_context(results)
    assert "[Extrait 1]" in context
    assert len(context) > 0


def test_retrieve_unknown_collection():
    """Vérifie l'erreur si la collection n'existe pas"""
    with pytest.raises(ValueError):
        retrieve("test", "collection_inexistante")