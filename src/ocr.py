"""
Module OCR : extraction de texte depuis des images ou des pages de PDF
scannees (sans couche de texte numerique), via Tesseract.
"""

from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def ocr_image(image: Image.Image, lang: str = "fra") -> str:
    """Extrait le texte d'une image PIL via Tesseract."""
    return pytesseract.image_to_string(image, lang=lang)


def ocr_pdf_pages(pdf_path: str | Path, lang: str = "fra", dpi: int = 200) -> list[str]:
    """
    Convertit chaque page d'un PDF en image puis applique l'OCR.
    Utilise pour les PDF 'scannes' qui ne contiennent aucune couche
    de texte exploitable directement.
    """
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    return [ocr_image(page, lang=lang) for page in pages]
