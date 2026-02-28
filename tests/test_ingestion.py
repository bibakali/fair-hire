"""
Tests unitaires pour ingestion.py
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_pdf, split_text, load_and_split


def test_split_text_basic():
    """Vérifie que le découpage fonctionne sur un texte simple"""
    texte = "Python " * 300  # Texte long artificiel
    chunks = split_text(texte, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_split_text_not_empty():
    """Vérifie qu'aucun morceau n'est vide"""
    texte = "Ceci est un test de découpage de texte. " * 50
    chunks = split_text(texte)
    assert all(len(c.strip()) > 0 for c in chunks)


def test_load_pdf_file_not_found():
    """Vérifie que l'erreur est levée si le fichier n'existe pas"""
    with pytest.raises(FileNotFoundError):
        load_pdf("fichier_inexistant.pdf")


def test_load_pdf_wrong_extension():
    """Vérifie que l'erreur est levée si ce n'est pas un PDF"""
    with pytest.raises(ValueError):
        load_pdf("document.txt")