"""Scraper de la FAQ facturation électronique d'impots.gouv.fr (DGFiP).

Une page liste présente les questions sous forme de cartes du Système de
Design de l'État ; chaque réponse vit sur sa propre page, visitée avec une
pause de politesse entre les requêtes.
"""

import time
from datetime import datetime

from bs4 import BeautifulSoup, Tag

from src.scraping.common import (
    FaqEntry,
    UnexpectedStructureError,
    clean_text,
    fetch_html,
)

BASE_URL = "https://www.impots.gouv.fr"
FAQ_LIST_URL = f"{BASE_URL}/professionnel/je-passe-la-facturation-electronique"
SOURCE_NAME = "DGFiP"
PAUSE_BETWEEN_REQUESTS_SECONDS = 1.0

# La question est le lien d'une carte ; le href pointe vers la page réponse.
QUESTION_LINK_SELECTOR = 'h3.fr-card__title > a[href^="/professionnel/questions/"]'
# Date de publication machine-readable sur la page réponse (format JJ/MM/AAAA).
PUBLISHED_DATE_SELECTOR = "p.datePublished span[data-published-date]"
# Corps de la réponse : premier bloc de contenu de l'article.
ANSWER_BODY_SELECTOR = "article[about] div.fr-col-12"


def scrape_dgfip() -> list[FaqEntry]:
    """Collecte les questions-réponses de la FAQ DGFiP.

    Fail-fast : la première page réponse illisible interrompt la collecte,
    plutôt que de renvoyer une liste partielle silencieuse.
    """
    list_html = fetch_html(FAQ_LIST_URL)
    list_soup = BeautifulSoup(list_html, "html.parser")

    links = list_soup.select(QUESTION_LINK_SELECTOR)
    if not links:
        raise UnexpectedStructureError(
            f"Aucune carte de question trouvée sur {FAQ_LIST_URL} "
            f"(sélecteur « {QUESTION_LINK_SELECTOR} ») : la mise en page a changé."
        )

    entries: list[FaqEntry] = []
    seen_urls: set[str] = set()
    for link in links:
        href = link.get("href")
        if not isinstance(href, str):
            continue
        answer_url = f"{BASE_URL}{href}"
        # Certaines cartes sont rendues en double (variantes d'affichage).
        if answer_url in seen_urls:
            continue
        seen_urls.add(answer_url)

        question = clean_text(link.get_text())
        if not question:
            raise UnexpectedStructureError(
                f"Carte sans texte de question sur {FAQ_LIST_URL} (lien {href})."
            )

        time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)
        answer_html = fetch_html(answer_url)
        answer_soup = BeautifulSoup(answer_html, "html.parser")

        entries.append(
            FaqEntry(
                question=question,
                reponse=_extract_answer(answer_soup, answer_url),
                source=SOURCE_NAME,
                date_heure_parution=_extract_published_date(answer_soup, answer_url),
                url=answer_url,
            )
        )

    return entries


def _extract_answer(answer_soup: BeautifulSoup, answer_url: str) -> str:
    """Extrait le corps de la réponse : paragraphes et éléments de listes.

    Le widget d'avis « Cet article vous a-t-il été utile ? » (bloc ``qr-rate``)
    est écarté pour ne garder que le contenu éditorial.
    """
    body = answer_soup.select_one(ANSWER_BODY_SELECTOR)
    if body is None:
        raise UnexpectedStructureError(
            f"Corps de réponse introuvable sur {answer_url} "
            f"(sélecteur « {ANSWER_BODY_SELECTOR} ») : la mise en page a changé."
        )

    lines: list[str] = []
    for element in body.find_all(["p", "li"]):
        if not isinstance(element, Tag) or element.find_parent(class_="qr-rate"):
            continue
        text = clean_text(element.get_text())
        if text:
            lines.append(f"- {text}" if element.name == "li" else text)

    if not lines:
        raise UnexpectedStructureError(
            f"Réponse vide sur {answer_url} : aucun paragraphe ni élément de liste."
        )
    return "\n".join(lines)


def _extract_published_date(
    answer_soup: BeautifulSoup, answer_url: str
) -> datetime | None:
    """Lit la date de publication ; ``None`` si le site ne l'affiche plus."""
    date_tag = answer_soup.select_one(PUBLISHED_DATE_SELECTOR)
    if date_tag is None:
        return None
    raw_date = date_tag.get("data-published-date")
    if not isinstance(raw_date, str):
        return None
    try:
        return datetime.strptime(raw_date, "%d/%m/%Y")
    except ValueError as exc:
        raise UnexpectedStructureError(
            f"Date de publication illisible sur {answer_url} : "
            f"« {raw_date} » n'est pas au format JJ/MM/AAAA."
        ) from exc
