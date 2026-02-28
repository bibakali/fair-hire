"""
Tests unitaires pour bias_detector.py
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bias_detector import (
    detect_gendered_words,
    detect_discriminatory_patterns,
    compute_bias_score,
    generate_suggestions,
    analyze,
    format_report
)


def test_detect_gendered_words_found():
    """Vérifie la détection de mots genrés"""
    text = "Nous cherchons un ninja rockstar ambitieux."
    found = detect_gendered_words(text)
    assert len(found) > 0
    assert any("ninja" in w for w in found)


def test_detect_gendered_words_clean():
    """Vérifie qu'une offre neutre ne déclenche rien"""
    text = "Nous cherchons un développeur Python expérimenté."
    found = detect_gendered_words(text)
    assert len(found) == 0


def test_detect_discriminatory_age():
    """Vérifie la détection de tranches d'âge"""
    text = "Candidat entre 25 et 35 ans requis."
    found = detect_discriminatory_patterns(text)
    assert len(found) > 0


def test_compute_bias_score_zero():
    """Vérifie que score = 0 si aucun biais"""
    score = compute_bias_score([], [], 100)
    assert score == 0.0


def test_compute_bias_score_positive():
    """Vérifie que le score augmente avec les biais"""
    score = compute_bias_score(["ninja (masculins)", "rockstar (masculins)"], [], 50)
    assert score > 0


def test_analyze_biased_text():
    """Vérifie l'analyse complète d'un texte biaisé"""
    text = "Cherchons ninja rockstar entre 25 et 35 ans, présentable."
    report = analyze(text)
    assert report.bias_score > 0
    assert len(report.gendered_words_found) > 0
    assert report.summary != ""


def test_analyze_clean_text():
    """Vérifie l'analyse d'un texte neutre"""
    text = "Nous recrutons un développeur Python motivé avec 3 ans d'expérience."
    report = analyze(text)
    assert report.bias_score == 0.0
    assert "✅" in report.summary


def test_format_report_contains_score():
    """Vérifie que le rapport formaté contient le score"""
    text = "Ninja rockstar ambitieux cherché."
    report = analyze(text)
    formatted = format_report(report)
    assert "Score de biais" in formatted
    assert "Suggestions" in formatted