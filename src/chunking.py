"""
Decoupage (chunking) du texte extrait en passages de taille geree,
avec un leger chevauchement pour ne pas couper une idee en deux.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(
    text: str,
    doc_id: str,
    max_chars: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """
    Decoupe un texte en chunks de longueur maximale `max_chars`
    caracteres, avec un chevauchement de `overlap` caracteres entre
    deux chunks consecutifs (pour eviter de perdre le contexte a la
    frontiere entre deux passages).
    """
    text = " ".join(text.split())  # normalise les espaces/retours a la ligne
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text_value = text[start:end]
        chunks.append(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text_value))
        chunk_id += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks
