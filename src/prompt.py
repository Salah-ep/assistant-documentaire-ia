"""
Construction du prompt envoye au LLM (prompt engineering) : instructions
systeme, contexte recupere par la recherche semantique, et question de
l'utilisateur. Structure volontairement explicite et commentee pour
servir de base de discussion (choix de formulation, limitation des
hallucinations, citation des sources).
"""

from __future__ import annotations

from .index_store import SearchResult

SYSTEM_INSTRUCTIONS = """Tu es un assistant documentaire. Reponds UNIQUEMENT a partir \
des extraits de documents fournis ci-dessous. Si l'information n'est pas presente \
dans ces extraits, dis clairement que tu ne sais pas plutot que d'inventer une reponse. \
Cite le numero du passage utilise entre crochets, par exemple [passage 2]."""


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """
    Assemble le prompt final a partir :
      - des instructions systeme (limiter les hallucinations, exiger
        des citations de source) ;
      - des passages recuperes par la recherche semantique (contexte) ;
      - de la question posee par l'utilisateur.

    C'est ce texte qui serait envoye a un LLM (via l'API choisie) dans
    `generate.py`. Le separer clairement permet de l'inspecter et de
    l'ajuster independamment du reste du pipeline (prompt engineering).
    """
    context_blocks = []
    for i, r in enumerate(results, start=1):
        context_blocks.append(f"[passage {i}] (source: {r.chunk.doc_id}, score={r.score:.2f})\n{r.chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(aucun passage pertinent trouve)"

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"### Extraits de documents\n{context}\n\n"
        f"### Question\n{question}\n\n"
        f"### Reponse"
    )
