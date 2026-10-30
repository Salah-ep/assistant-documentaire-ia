"""
Vectorisation (embeddings) des chunks de texte pour la recherche
semantique.

Utilise les vecteurs de mots pre-entraines du modele spaCy francais
`fr_core_news_md` (vecteurs de type floret, dimension 300), moyennes sur
les tokens d'un passage pour obtenir un vecteur de document. C'est une
alternative legere et 100% locale a des embeddings de type
sentence-transformers (qui necessitent un telechargement depuis
Hugging Face, indisponible dans certains environnements sans acces
reseau complet).
"""

from __future__ import annotations

import numpy as np
import spacy

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("fr_core_news_md")
    return _NLP


def embed_text(text: str) -> np.ndarray:
    """Retourne le vecteur semantique (300 dimensions) d'un texte."""
    nlp = get_nlp()
    doc = nlp(text)
    vec = doc.vector.astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_texts(texts: list[str]) -> np.ndarray:
    """Vectorise une liste de textes, retourne une matrice (n, 300)."""
    return np.vstack([embed_text(t) for t in texts]) if texts else np.zeros((0, 300), dtype="float32")
