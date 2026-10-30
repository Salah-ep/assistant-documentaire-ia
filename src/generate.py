"""
Etape de generation de la reponse (la "G" de RAG), concue comme une
interface interchangeable (deux backends) :

  - ExtractiveGenerator (par defaut) : ne depend d'aucune API externe,
    fonctionne entierement hors-ligne. Elle construit une reponse en
    selectionnant, dans les passages recuperes, les phrases les plus
    proches lexicalement de la question. Utile en environnement sans
    acces a une API LLM (ex. sandbox sans sortie reseau vers les
    fournisseurs de LLM), et pour avoir une baseline toujours testable.

  - LLMGenerator (optionnel) : envoie le prompt construit dans
    `prompt.py` a un LLM via une API compatible OpenAI (OpenAI,
    ou tout serveur local type Ollama expose en mode OpenAI-compatible).
    Necessite une cle API (variable d'environnement) et un acces
    reseau sortant vers le fournisseur choisi - non exerce dans cet
    environnement de developpement sandboxe (voir DOCUMENTATION.md).

Le pipeline (pipeline.py) choisit le backend via un parametre, ce qui
permet de brancher un vrai LLM des que les identifiants sont
disponibles, sans changer le reste du code.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from .index_store import SearchResult


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, results: list[SearchResult], prompt: str) -> str:
        ...


class ExtractiveGenerator(Generator):
    """Backend par defaut, 100% local, sans dependance a une API LLM."""

    def _split_sentences(self, text: str) -> list[str]:
        # Decoupage simple par ponctuation forte - suffisant pour des
        # passages courts issus du chunking.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    def _score_sentence(self, sentence: str, question_words: set[str]) -> int:
        words = set(re.findall(r"\w+", sentence.lower()))
        return len(words & question_words)

    def generate(self, question: str, results: list[SearchResult], prompt: str) -> str:
        if not results:
            return "Je ne trouve pas d'information pertinente dans les documents indexes pour repondre a cette question."

        question_words = set(re.findall(r"\w+", question.lower()))
        scored: list[tuple[int, str, int]] = []  # (score, phrase, numero de passage)
        for i, r in enumerate(results, start=1):
            for sentence in self._split_sentences(r.chunk.text):
                score = self._score_sentence(sentence, question_words)
                if score > 0:
                    scored.append((score, sentence, i))

        if not scored:
            # Aucun recouvrement lexical direct : on retourne le passage le mieux classe tel quel.
            best = results[0]
            return f"D'apres le passage le plus pertinent trouve [passage 1] : {best.chunk.text.strip()}"

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored[:3]
        answer_parts = [f"{sentence.strip()} [passage {num}]" for _, sentence, num in top_sentences]
        return " ".join(answer_parts)


class LLMGenerator(Generator):
    """
    Backend optionnel : appel a un LLM via une API compatible OpenAI.
    Necessite OPENAI_API_KEY (ou OPENAI_BASE_URL pour un serveur local
    type Ollama) dans l'environnement. Non teste en direct dans ce
    sandbox de developpement (pas d'acces reseau sortant vers les
    fournisseurs de LLM) - a valider sur une machine avec acces internet.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def generate(self, question: str, results: list[SearchResult], prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "LLMGenerator necessite la variable d'environnement OPENAI_API_KEY "
                "(ou un serveur compatible via OPENAI_BASE_URL)."
            )
        import requests  # import local : dependance optionnelle, seulement si ce backend est utilise

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_generator(backend: str = "extractive") -> Generator:
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "llm":
        return LLMGenerator()
    raise ValueError(f"Backend de generation inconnu : {backend}")
