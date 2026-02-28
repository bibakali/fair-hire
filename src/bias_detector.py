"""
bias_detector.py
Détection de biais dans les offres d'emploi
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------
# Dictionnaires de biais — enrichissables facilement
# ---------------------------------------------------------------

GENDERED_WORDS = {
    "masculins": [
        "ninja", "rockstar", "guru", "wizard", "hacker",
        "ambitieux", "combatif", "dominant", "compétitif",
        "indépendant", "confiant", "assertif", "agressif"
    ],
    "feminins": [
        "collaboratif", "empathique", "doux", "attentionné",
        "discret", "patient", "bienveillant"
    ]
}

DISCRIMINATORY_PATTERNS = [
    r"\d{2}\s*[-–\s*et\s*]\d{2}\s*ans",  # 25-35 ans ou 25 et 35 ans
    r"\d{2}\s*ans",                        # "25 ans" seul
    r"jeune",
    r"apparence",
    r"présentable",
    r"photos?",
    r"disponible\s*immédiatement",
]

INCLUSIVE_ALTERNATIVES = {
    "ninja": "expert",
    "rockstar": "développeur talentueux",
    "guru": "spécialiste",
    "wizard": "expert",
    "hacker": "développeur créatif",
    "ambitieux": "motivé",
    "combatif": "déterminé",
    "dominant": "leadership",
    "jeune": "junior",
}


# ---------------------------------------------------------------
# Dataclass pour structurer les résultats
# ---------------------------------------------------------------

@dataclass
class BiasReport:
    gendered_words_found: list[str] = field(default_factory=list)
    discriminatory_patterns_found: list[str] = field(default_factory=list)
    bias_score: float = 0.0          # 0 = neutre, 1 = très biaisé
    suggestions: dict = field(default_factory=dict)
    rewritten_excerpt: str = ""
    summary: str = ""


# ---------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------

def detect_gendered_words(text: str) -> list[str]:
    """Détecte les mots genrés dans le texte."""
    text_lower = text.lower()
    found = []
    for genre, words in GENDERED_WORDS.items():
        for word in words:
            if word in text_lower:
                found.append(f"{word} ({genre})")
    return found


def detect_discriminatory_patterns(text: str) -> list[str]:
    """Détecte les patterns discriminatoires via regex."""
    text_lower = text.lower()
    found = []
    for pattern in DISCRIMINATORY_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            found.append(pattern)
    return found


def compute_bias_score(
    gendered: list[str],
    discriminatory: list[str],
    total_words: int
) -> float:
    """
    Calcule un score de biais entre 0 et 1.
    Plus le score est élevé, plus l'offre est biaisée.
    """
    if total_words == 0:
        return 0.0

    raw_score = (len(gendered) * 2 + len(discriminatory) * 3) / total_words * 100
    return round(min(raw_score, 1.0), 4)


def generate_suggestions(gendered_words: list[str]) -> dict:
    """Génère des suggestions de remplacement pour les mots biaisés."""
    suggestions = {}
    for entry in gendered_words:
        word = entry.split(" (")[0]  # retire "(masculins)" ou "(feminins)"
        if word in INCLUSIVE_ALTERNATIVES:
            suggestions[word] = INCLUSIVE_ALTERNATIVES[word]
        else:
            suggestions[word] = "⚠️ À reformuler (pas d'alternative automatique)"
    return suggestions


def analyze(text: str) -> BiasReport:
    """
    Analyse complète d'un texte pour détecter les biais.

    Args:
        text: Texte de l'offre d'emploi

    Returns:
        BiasReport avec tous les résultats
    """
    report = BiasReport()

    # Détections
    report.gendered_words_found = detect_gendered_words(text)
    report.discriminatory_patterns_found = detect_discriminatory_patterns(text)

    # Score
    total_words = len(text.split())
    report.bias_score = compute_bias_score(
        report.gendered_words_found,
        report.discriminatory_patterns_found,
        total_words
    )

    # Suggestions
    report.suggestions = generate_suggestions(report.gendered_words_found)

    # Résumé lisible
    if report.bias_score == 0:
        report.summary = "✅ Aucun biais détecté. L'offre semble inclusive."
    elif report.bias_score < 0.05:
        report.summary = f"⚠️ Biais faibles détectés ({len(report.gendered_words_found)} mots genrés)."
    else:
        report.summary = f"🚨 Biais significatifs détectés — reformulation recommandée."

    return report


def format_report(report: BiasReport) -> str:
    """Formate le rapport pour affichage."""
    lines = [
        "=" * 50,
        "RAPPORT D'ANALYSE DES BIAIS",
        "=" * 50,
        f"\n📊 Score de biais : {report.bias_score}",
        f"📝 Résumé : {report.summary}",
    ]

    if report.gendered_words_found:
        lines.append(f"\n🔍 Mots genrés détectés :")
        for w in report.gendered_words_found:
            lines.append(f"  - {w}")

    if report.discriminatory_patterns_found:
        lines.append(f"\n🚫 Patterns discriminatoires :")
        for p in report.discriminatory_patterns_found:
            lines.append(f"  - {p}")

    if report.suggestions:
        lines.append(f"\n💡 Suggestions de remplacement :")
        for original, alternative in report.suggestions.items():
            lines.append(f"  - '{original}' → '{alternative}'")

    lines.append("=" * 50)
    return "\n".join(lines)


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    offre = """
    Nous recherchons un ninja du code, rockstar et ambitieux,
    entre 25 et 35 ans, présentable et disponible immédiatement.
    Vous êtes indépendant, combatif et aimez les défis techniques.
    """

    report = analyze(offre)
    print(format_report(report))