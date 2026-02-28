"""
Tests unitaires pour agent.py
On mocke les appels externes (Mistral, ChromaDB)
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import (
    tool_detect_bias,
    tool_retrieve_context,
    FairHireResult,
    run_pipeline
)


def test_fair_hire_result_defaults():
    """Vérifie les valeurs par défaut du dataclass"""
    result = FairHireResult()
    assert result.status == "pending"
    assert result.bias_score == 0.0
    assert result.error == ""


def test_tool_detect_bias_biased():
    """Vérifie la détection de biais via l'outil"""
    text = "Cherchons ninja rockstar ambitieux entre 25 et 35 ans."
    report, score = tool_detect_bias(text)
    assert score > 0
    assert "Score de biais" in report


def test_tool_detect_bias_clean():
    """Vérifie qu'un texte neutre a un score de 0"""
    text = "Nous recrutons un développeur Python motivé."
    report, score = tool_detect_bias(text)
    assert score == 0.0


def test_run_pipeline_file_not_found():
    """Vérifie que le pipeline gère les fichiers manquants"""
    result = run_pipeline("cv_inexistant.pdf", "job_inexistant.pdf")
    assert result.status == "error"
    assert result.error != ""


def test_tool_retrieve_context_mocked():
    """Vérifie le retrieve avec mock ChromaDB"""
    with patch("src.agent.retrieve") as mock_retrieve, \
         patch("src.agent.format_context") as mock_format:

        mock_retrieve.return_value = [{"text": "Python dev", "score": 0.9, "metadata": {}}]
        mock_format.return_value = "Python dev"

        context = tool_retrieve_context("compétences", "cv_current")
        assert context == "Python dev"