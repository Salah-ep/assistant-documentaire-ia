"""
Index vectoriel (FAISS) stockant les embeddings des chunks de documents
et permettant la recherche semantique par similarite cosinus
(implementee via produit scalaire sur des vecteurs normalises).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dim: int = 300):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # produit scalaire = cosinus (vecteurs normalises)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if len(self.chunks) == 0:
            return []
        query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str | Path, dim: int = 300) -> "VectorStore":
        path = Path(path)
        store = cls(dim=dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
