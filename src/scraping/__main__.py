"""Point d'entrée de la collecte : ``python -m src.scraping``.

Lance chaque scraper en isolant ses échecs (une source en panne ne fait pas
perdre les autres), sauvegarde le tout et affiche un compte rendu — c'est ce
que cron enverra par mail. Code de sortie : 0 si au moins une source a
fonctionné, 1 si toutes ont échoué.
"""

import sys
from collections.abc import Callable

from src.scraping.common import FaqEntry, ScrapingError
from src.scraping.dgfip import scrape_dgfip
from src.scraping.lecoindesentrepreneurs import scrape_lecoin_des_entrepreneurs
from src.scraping.storage import CSV_PATH, save_entries

SCRAPERS: list[tuple[str, Callable[[], list[FaqEntry]]]] = [
    ("DGFiP", scrape_dgfip),
    ("LeCoinDesEntrepreneurs", scrape_lecoin_des_entrepreneurs),
]


def main() -> int:
    """Collecte toutes les sources, sauvegarde, affiche le compte rendu."""
    entries: list[FaqEntry] = []
    failures = 0

    # Seule ScrapingError est rattrapée : les scrapers traduisent déjà tous
    # leurs échecs attendus ; toute autre exception est un bug et doit remonter.
    for name, scrape in SCRAPERS:
        try:
            collected = scrape()
        except ScrapingError as exc:
            failures += 1
            print(f"{name} : ERREUR — {exc}", file=sys.stderr)
        else:
            entries.extend(collected)
            print(f"{name} : {len(collected)} entrées collectées")

    if failures == len(SCRAPERS):
        print("Toutes les sources ont échoué, rien à sauvegarder.", file=sys.stderr)
        return 1

    added, skipped = save_entries(entries)
    print(
        f"Sauvegarde : {added} nouvelles entrées, {skipped} déjà connues → {CSV_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
