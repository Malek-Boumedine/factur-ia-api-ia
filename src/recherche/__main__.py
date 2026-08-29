"""CLI du POC de recherche sémantique.

Usage : ``uv run python -m src.recherche "ma question en langage naturel"``.
Affiche les meilleures entrées du corpus avec leur score de similarité, leur
source et un extrait de la réponse, plus les temps de chargement et de
recherche — des données de démonstration, pas du logging.
"""

import argparse
import textwrap
import time

from src.recherche.engine import SearchEngine


def main() -> None:
    """Encode le corpus, exécute la requête et affiche les résultats."""
    parser = argparse.ArgumentParser(
        prog="python -m src.recherche",
        description="Recherche sémantique dans la FAQ réglementaire (data/faq.csv).",
    )
    parser.add_argument("question", help="question en langage naturel")
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="nombre de résultats à afficher (défaut : 3)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    engine = SearchEngine()
    ready = time.perf_counter()
    results = engine.search(args.question, top_k=args.top)
    done = time.perf_counter()

    print(
        f"Corpus : {len(engine.documents)} entrées — modèle chargé et corpus "
        f"encodé en {ready - start:.1f} s, recherche en {(done - ready) * 1000:.0f} ms"
    )
    print(f"Requête : {args.question}\n")
    for rank, result in enumerate(results, start=1):
        excerpt = textwrap.shorten(result.document.reponse, width=280)
        print(f"{rank}. [score {result.score:.3f}] {result.document.question}")
        print(f"   Source : {result.document.source} — {result.document.url}")
        print(f"   {excerpt}\n")


if __name__ == "__main__":
    main()
