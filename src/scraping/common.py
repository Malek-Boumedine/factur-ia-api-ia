"""Contrat de sortie des scrapers de FAQ et briques communes (erreurs, HTTP, texte).

Chaque scraper renvoie une ``list[FaqEntry]`` et signale ses échecs via les
exceptions définies ici : réseau ou HTTP en erreur → ``SourceUnavailableError``,
sélecteur qui ne trouve plus rien (le site a changé de mise en page) →
``UnexpectedStructureError``. Jamais de liste partielle silencieuse.
"""

from datetime import datetime
from typing import TypedDict

import httpx

# User-Agent identifiable : la source doit pouvoir savoir qui la consulte.
USER_AGENT = "factur-ia-scraper/0.1 (contact: factur-ia@gmail.com)"
REQUEST_TIMEOUT_SECONDS = 10.0


class FaqEntry(TypedDict):
    """Une question-réponse collectée, structure commune à toutes les sources.

    ``date_heure_scraping`` sera ajoutée par le module de stockage, pas ici.
    """

    question: str
    reponse: str
    source: str
    date_heure_parution: datetime | None
    url: str


class ScrapingError(Exception):
    """Erreur de base de la collecte des FAQ."""


class SourceUnavailableError(ScrapingError):
    """La source est injoignable : timeout, erreur réseau ou code HTTP d'erreur."""


class UnexpectedStructureError(ScrapingError):
    """La page n'a plus la structure attendue : un sélecteur ne trouve rien.

    C'est la limite connue du scraping — le message dit précisément quel
    élément manque et sur quelle URL, pour rendre le diagnostic immédiat.
    """


def fetch_html(url: str) -> str:
    """Télécharge une page HTML, traduit tout échec en ``SourceUnavailableError``."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SourceUnavailableError(
            f"Réponse HTTP {exc.response.status_code} pour {url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise SourceUnavailableError(f"Échec réseau pour {url} : {exc}") from exc
    html: str = response.text
    return html


def clean_text(text: str) -> str:
    """Normalise les espaces (insécables compris) en conservant la typographie.

    BeautifulSoup décode déjà les entités HTML ; les apostrophes et guillemets
    typographiques sont conservés tels quels, valides en français.
    """
    return " ".join(text.split())
