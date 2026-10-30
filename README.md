# Assistant Documentaire IA

Assistant documentaire interrogeant une base de documents PDF en langage
naturel, via une architecture **RAG** (Retrieval-Augmented Generation)
avec support **OCR** pour les documents scannes (sans couche de texte
numerique).

> Pour une explication detaillee, pedagogique, de chaque etape du
> pipeline (a destination de quelqu'un qui decouvre le projet), voir
> [`DOCUMENTATION.md`](./DOCUMENTATION.md).

## Fonctionnalites

- **Ingestion mixte** : detecte automatiquement si un PDF contient une
  couche de texte exploitable (extraction directe, rapide) ou s'il
  s'agit d'un document scanne/image (bascule automatique sur l'OCR,
  Tesseract, francais + anglais).
- **Vectorisation / recherche semantique** : chaque passage de document
  est transforme en vecteur (embeddings de mots spaCy, francais,
  300 dimensions) et indexe dans un index vectoriel FAISS pour une
  recherche par similarite cosinus.
- **Prompt engineering explicite** : le prompt envoye au modele de
  generation est construit dans un module dedie (`src/prompt.py`),
  avec des instructions visant a limiter les hallucinations et a citer
  les passages sources.
- **Backend de generation interchangeable** : un mode extractif par
  defaut (100% local, aucune dependance a une API) et un mode LLM
  (API compatible OpenAI) pret a etre branche des qu'une cle API est
  disponible.

## Architecture

```
PDF --> loader.py (texte natif ou OCR) --> chunking.py --> embeddings.py --> index_store.py (FAISS)
                                                                                    |
Question --> embeddings.py -----------------------------------------------> recherche (top-k)
                                                                                    |
                                                                              prompt.py --> generate.py --> Reponse
```

## Installation

Dependances systeme (Debian/Ubuntu) :

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng poppler-utils
```

Dependances Python :

```bash
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl
```

## Utilisation

Generer des documents de demonstration (un PDF numerique + un PDF
"scanne" avec le meme contenu, pour verifier concretement le chemin
OCR) puis lancer la demo :

```bash
python3 scripts/generate_test_docs.py
python3 scripts/demo.py
```

Utilisation programmatique :

```python
from src.pipeline import DocumentAssistant

assistant = DocumentAssistant(generator_backend="extractive")  # ou "llm"
assistant.ingest_pdf("data/mon_document.pdf")

resultat = assistant.ask("Quelle est la procedure de maintenance recommandee ?")
print(resultat["answer"])
```

## Activer un vrai LLM pour la generation

Par defaut, la generation est **extractive** (aucune dependance
externe). Pour utiliser un vrai LLM via une API compatible OpenAI :

```bash
export OPENAI_API_KEY="sk-..."
```

```python
assistant = DocumentAssistant(generator_backend="llm")
```

## Limites connues

- Les embeddings de mots spaCy (vecteurs moyennes par passage) sont
  plus simples que des embeddings de phrases neuronaux (type
  sentence-transformers) : ils capturent bien la similarite lexicale
  et thematique, moins bien les nuances syntaxiques fines.
- L'OCR (Tesseract) est sensible a la qualite du scan (resolution,
  contraste, inclinaison) : des documents scannes de mauvaise qualite
  degraderont la qualite de l'extraction.
- Le backend extractif ne "redige" pas une reponse a proprement parler :
  il selectionne et assemble les phrases les plus pertinentes des
  passages retrouves.
