"""
ingestion.py
Lecture et découpage des documents PDF (CVs et offres d'emploi)
"""

import os
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier doit être un PDF : {file_path}")

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    text = ""
    doc = fitz.open(file_path)

    try:
        for page_num, page in enumerate(doc):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        n_pages = len(doc)
    finally:
        doc.close()

    if not text.strip():
        raise ValueError(f"Aucun texte extrait du PDF : {file_path}")

    print(f"✅ PDF chargé : {path.name} ({n_pages} pages, {len(text)} caractères)")
    return text

def load_txt(file_path: str) -> str:
    """Lit un fichier texte brut."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"Fichier texte vide : {file_path}")

    print(f"✅ Texte chargé : {path.name} ({len(text)} caractères)")
    return text

def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """
    Découpe le texte en morceaux pour la vectorisation.
    
    Args:
        text: Texte brut à découper
        chunk_size: Taille de chaque morceau (en caractères)
        chunk_overlap: Chevauchement entre morceaux (évite de couper les idées)
        
    Returns:
        Liste de morceaux de texte
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_text(text)
    print(f"✅ Texte découpé : {len(chunks)} morceaux (chunk_size={chunk_size})")
    return chunks


def load_and_split(file_path: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """
    Pipeline complet : charge un PDF ou TXT et le découpe.
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        text = load_pdf(file_path)
    elif path.suffix.lower() == ".txt":
        text = load_txt(file_path)
    else:
        raise ValueError(f"Format non supporté : {path.suffix}. Utilisez PDF ou TXT.")

    chunks = split_text(text, chunk_size, chunk_overlap)
    return chunks


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py <chemin_vers_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    chunks = load_and_split(pdf_path)
    
    print(f"\n--- Aperçu du premier morceau ---")
    print(chunks[0])
    print(f"\n--- Aperçu du dernier morceau ---")
    print(chunks[-1])