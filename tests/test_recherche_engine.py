"""Tests de la partie pure du moteur de recherche sémantique (POC).

Seuls la lecture du CSV et le classement par similarité sont testés : le
chargement du modèle d'embeddings exigerait un téléchargement, interdit par la
garde réseau de la suite — non testé, assumé pour un POC.
"""

from pathlib import Path

import numpy as np
from src.recherche.engine import FaqDocument, load_documents, rank_by_similarity

CSV_CONTENT = (
    "question,reponse,source,date_heure_parution,url,date_heure_scraping\n"
    "Quand ?,Bientôt.,DGFiP,,https://exemple.fr/quand,2026-01-01T00:00:00\n"
    'Qui ?,"Tout le monde, sans exception.",DGFiP,,https://exemple.fr/qui,'
    "2026-01-01T00:00:00\n"
)


def test_load_documents_lit_les_champs_utiles(tmp_path: Path) -> None:
    csv_path = tmp_path / "faq.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    documents = load_documents(csv_path)

    assert documents == [
        FaqDocument(
            question="Quand ?",
            reponse="Bientôt.",
            source="DGFiP",
            url="https://exemple.fr/quand",
        ),
        FaqDocument(
            question="Qui ?",
            reponse="Tout le monde, sans exception.",
            source="DGFiP",
            url="https://exemple.fr/qui",
        ),
    ]


def test_load_documents_sur_le_csv_reel() -> None:
    """Le contrat avec le scraping est le fichier : le CSV du dépôt doit se charger."""
    documents = load_documents()

    assert len(documents) >= 1
    assert all(doc.question and doc.reponse for doc in documents)


def test_rank_by_similarity_classe_par_score_decroissant() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]],  # scores : 0.0, 1.0, 0.6
        dtype=np.float32,
    )

    ranked = rank_by_similarity(query, documents, top_k=3)

    assert [index for index, _ in ranked] == [1, 2, 0]
    scores = [score for _, score in ranked]
    assert scores[0] == 1.0
    assert scores == sorted(scores, reverse=True)


def test_rank_by_similarity_respecte_top_k() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)

    ranked = rank_by_similarity(query, documents, top_k=1)

    assert ranked == [(0, 1.0)]
