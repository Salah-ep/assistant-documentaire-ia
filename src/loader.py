"""
Module d'ingestion de documents : charge un PDF, detecte s'il contient
une couche de texte numerique exploitable ou s'il s'agit d'un document
scanne (image), et extrait le texte en consequence (texte natif ou OCR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from . import ocr

# Seuil (nombre de caracteres) en dessous duquel on considere qu'une page
# n'a pas de couche de texte exploitable et qu'il faut passer par l'OCR.
MIN_CHARS_PER_PAGE = 20


@dataclass
class LoadedDocument:
    source_path: str
    pages: list[str]
    used_ocr: bool

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)


def _extract_native_text(pdf_path: str | Path) -> list[str]:
    """Extrait le texte natif (deja numerique) de chaque page d'un PDF."""
    doc = fitz.open(str(pdf_path))
    try:
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()


def load_pdf(pdf_path: str | Path, lang: str = "fra") -> LoadedDocument:
    """
    Charge un PDF :
      1. tente d'abord l'extraction de texte natif (rapide, fiable) ;
      2. si une page n'a (quasi) pas de texte natif, bascule sur l'OCR
         pour cette page precise (cas d'un document scanne / image).
    """
    native_pages = _extract_native_text(pdf_path)
    needs_ocr = any(len(p.strip()) < MIN_CHARS_PER_PAGE for p in native_pages)

    if not needs_ocr:
        return LoadedDocument(source_path=str(pdf_path), pages=native_pages, used_ocr=False)

    # Au moins une page est vide/illisible en texte natif -> on refait
    # tout le document en OCR pour rester simple et coherent.
    ocr_pages = ocr.ocr_pdf_pages(pdf_path, lang=lang)
    return LoadedDocument(source_path=str(pdf_path), pages=ocr_pages, used_ocr=True)
