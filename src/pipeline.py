"""
Orchestration du pipeline RAG complet :

  Ingestion :  PDF -> (texte natif ou OCR) -> chunking -> embeddings -> index vectoriel
  Interrogation : question -> embedding -> recherche semantique (top-k) -> prompt -> generation
"""

from __future__ import annotations

from pathlib import Path

from . import loader
from .chunking import chunk_text
from .embeddings import embed_text, embed_texts
from .generate import Generator, get_generator
from .index_store import VectorStore
from .prompt import build_prompt


class DocumentAssistant:
    def __init__(self, generator_backend: str = "extractive"):
        self.store = VectorStore(dim=300)
        self.generator: Generator = get_generator(generator_backend)
        self.ingested_docs: list[dict] = []  # trace pour la documentation / le debug

    def ingest_pdf(self, pdf_path: str | Path, lang: str = "fra") -> None:
        doc_id = Path(pdf_path).name
        loaded = loader.load_pdf(pdf_path, lang=lang)
        chunks = chunk_text(loaded.full_text, doc_id=doc_id)
        vectors = embed_texts([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.ingested_docs.append(
            {
                "doc_id": doc_id,
                "used_ocr": loaded.used_ocr,
                "n_pages": len(loaded.pages),
                "n_chunks": len(chunks),
            }
        )

    def ask(self, question: str, top_k: int = 4) -> dict:
        query_vector = embed_text(question)
        results = self.store.search(query_vector, top_k=top_k)
        prompt = build_prompt(question, results)
        answer = self.generator.generate(question, results, prompt)
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "sources": [
                {"doc_id": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "score": r.score, "text": r.chunk.text}
                for r in results
            ],
        }
