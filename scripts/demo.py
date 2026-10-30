"""
Demo de bout en bout : ingestion d'un document numerique ET d'un
document scanne (pour exercer reellement le chemin OCR), puis quelques
questions posees a l'assistant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import DocumentAssistant  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    assistant = DocumentAssistant(generator_backend="extractive")

    print("=== Ingestion ===")
    for pdf_name in ["rapport_numerique.pdf", "rapport_scanne.pdf"]:
        assistant.ingest_pdf(DATA_DIR / pdf_name)
    for doc in assistant.ingested_docs:
        print(f"  - {doc['doc_id']}: OCR={'oui' if doc['used_ocr'] else 'non'}, "
              f"{doc['n_pages']} page(s), {doc['n_chunks']} chunk(s)")

    questions = [
        "Pourquoi les capteurs de pression de la ligne B se encrassent-ils ?",
        "A quelle frequence faut-il nettoyer les capteurs ?",
    ]

    print("\n=== Questions ===")
    for q in questions:
        result = assistant.ask(q)
        print(f"\nQ: {q}")
        print(f"R: {result['answer']}")
        print("Sources utilisees:")
        for s in result["sources"]:
            print(f"  - {s['doc_id']} (chunk {s['chunk_id']}, score={s['score']:.2f})")


if __name__ == "__main__":
    main()
