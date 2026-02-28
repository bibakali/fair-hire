"""
Tests unitaires pour generator.py
On teste la logique sans appeler Mistral (trop lent pour les tests)
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import build_prompt


def test_build_prompt_general():
    """Vérifie que le prompt général est bien construit"""
    prompt = build_prompt("Qui est le candidat ?", "Jean Dupont, développeur.", mode="general")
    assert "Jean Dupont" in prompt
    assert "Qui est le candidat" in prompt


def test_build_prompt_matching():
    """Vérifie que le prompt matching contient les bons éléments"""
    prompt = build_prompt("Analyse le matching", "CV ici", mode="matching")
    assert "CV ici" in prompt
    assert "expert RH" in prompt


def test_build_prompt_bias():
    """Vérifie que le prompt bias est bien construit"""
    prompt = build_prompt("Détecte les biais", "Offre ici", mode="bias")
    assert "Offre ici" in prompt
    assert "biais" in prompt


def test_build_prompt_contains_context():
    """Vérifie que le contexte est toujours inclus dans le prompt"""
    context = "Contexte très important"
    prompt = build_prompt("Question ?", context)
    assert context in prompt


def test_generate_connection_error():
    """Vérifie que l'erreur de connexion est bien gérée"""
    from unittest.mock import patch
    from src.generator import generate

    with patch("src.generator.OLLAMA_HOST", "http://localhost:99999"):
        with pytest.raises((ConnectionError, RuntimeError, TimeoutError)):
            generate("test", "contexte test")