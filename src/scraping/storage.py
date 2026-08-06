"""Sauvegarde des FAQ collectées dans un CSV unique, avec déduplication.

Le CSV est sa propre mémoire : à chaque sauvegarde, les entrées déjà présentes
sont relues et seules les nouveautés sont ajoutées en fin de fichier. Une
réponse modifiée compte comme une nouveauté (l'empreinte porte sur le couple
question + réponse), elle apparaît donc avec sa nouvelle date de scraping.
"""

import csv
from datetime import datetime
from pathlib import Path

from src.scraping.common import FaqEntry

# Ancré sur la racine du projet : un chemin relatif au cwd casserait sous cron.
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "faq.csv"

FIELDNAMES = [
    "question",
    "reponse",
    "source",
    "date_heure_parution",
    "url",
    "date_heure_scraping",
]


def save_entries(entries: list[FaqEntry]) -> tuple[int, int]:
    """Ajoute au CSV les entrées inconnues, renvoie (ajoutées, déjà connues).

    L'empreinte d'une entrée est le tuple (question, réponse) : elle capte les
    questions nouvelles comme les réponses corrigées. Le timestamp de collecte
    est pris une seule fois pour tout le lot.
    """
    known = _read_known_fingerprints()
    scraped_at = datetime.now().astimezone().isoformat()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()

    added = 0
    skipped = 0
    with CSV_PATH.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for entry in entries:
            fingerprint = (entry["question"], entry["reponse"])
            if fingerprint in known:
                skipped += 1
                continue
            # Ajout au fil de l'eau : dédoublonne aussi au sein du lot courant.
            known.add(fingerprint)
            published = entry["date_heure_parution"]
            writer.writerow(
                {
                    "question": entry["question"],
                    "reponse": entry["reponse"],
                    "source": entry["source"],
                    "date_heure_parution": (
                        published.isoformat() if published is not None else ""
                    ),
                    "url": entry["url"],
                    "date_heure_scraping": scraped_at,
                }
            )
            added += 1
    return added, skipped


def _read_known_fingerprints() -> set[tuple[str, str]]:
    """Relit le CSV existant et renvoie les empreintes (question, réponse)."""
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return {(row["question"], row["reponse"]) for row in reader}
