"""
ingestion.py
Lecture et découpage des documents PDF (CVs et offres d'emploi)
"""

import os
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    """
    Lit un fichier PDF et retourne son contenu texte brut.
    
    Args:
        file_path: Chemin vers le fichier PDF
        
    Returns:
        Texte extrait du PDF
    """
    path = Path(file_path)
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier doit être un PDF : {file_path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    
    doc = fitz.open(file_path)
    text = ""
    
    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    
    doc.close()
    
    if not text.strip():
        raise ValueError(f"Aucun texte extrait du PDF : {file_path}")
    
    print(f"✅ PDF chargé : {path.name} ({len(doc)} pages, {len(text)} caractères)")
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
    Pipeline complet : charge un PDF et le découpe.
    Fonction principale à appeler depuis les autres modules.
    
    Args:
        file_path: Chemin vers le fichier PDF
        chunk_size: Taille des morceaux
        chunk_overlap: Chevauchement
        
    Returns:
        Liste de morceaux de texte prêts à être vectorisés
    """
    text = load_pdf(file_path)
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