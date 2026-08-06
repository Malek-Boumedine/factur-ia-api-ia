"""Scraper du dossier questions-réponses de lecoindesentrepreneurs.fr.

Page WordPress unique : chaque question est un ``h2`` du corps de l'article,
sa réponse est constituée des paragraphes et listes qui le suivent jusqu'au
``h2`` suivant. Les encarts publicitaires et boîtes à outils insérés au milieu
du contenu sont écartés en ne retenant que les blocs éditoriaux WordPress.
"""

from datetime import datetime

from bs4 import BeautifulSoup, Tag

from src.scraping.common import (
    FaqEntry,
    UnexpectedStructureError,
    clean_text,
    fetch_html,
)

FAQ_URL = (
    "https://www.lecoindesentrepreneurs.fr/questions-reponses-facturation-electronique/"
)
SOURCE_NAME = "LeCoinDesEntrepreneurs"

# Conteneur du corps de l'article (thème WordPress du site).
CONTENT_SELECTOR = "div.entry"
# Chaque question du dossier est un titre de section ancré (id="question-N").
QUESTION_HEADING_SELECTOR = "h2.wp-block-heading"
# Date de publication machine-readable (ISO 8601) dans les métadonnées Open Graph.
PUBLISHED_META_SELECTOR = 'meta[property="article:published_time"]'


def scrape_lecoin_des_entrepreneurs() -> list[FaqEntry]:
    """Collecte les questions-réponses du dossier facturation électronique."""
    html = fetch_html(FAQ_URL)
    soup = BeautifulSoup(html, "html.parser")

    content = soup.select_one(CONTENT_SELECTOR)
    if content is None:
        raise UnexpectedStructureError(
            f"Corps de l'article introuvable sur {FAQ_URL} "
            f"(sélecteur « {CONTENT_SELECTOR} ») : la mise en page a changé."
        )

    headings = content.select(QUESTION_HEADING_SELECTOR)
    if not headings:
        raise UnexpectedStructureError(
            f"Aucun titre de question trouvé sur {FAQ_URL} "
            f"(sélecteur « {QUESTION_HEADING_SELECTOR} ») : la mise en page a changé."
        )

    published_date = _extract_published_date(soup)

    entries: list[FaqEntry] = []
    for heading in headings:
        question = clean_text(heading.get_text())
        if not question:
            raise UnexpectedStructureError(
                f"Titre de question vide sur {FAQ_URL} (id « {heading.get('id')} »)."
            )
        anchor = heading.get("id")
        entries.append(
            FaqEntry(
                question=question,
                reponse=_extract_answer(heading, question),
                source=SOURCE_NAME,
                date_heure_parution=published_date,
                url=f"{FAQ_URL}#{anchor}" if isinstance(anchor, str) else FAQ_URL,
            )
        )

    return entries


def _extract_answer(heading: Tag, question: str) -> str:
    """Rassemble la réponse : les blocs éditoriaux entre ce ``h2`` et le suivant.

    Seuls les paragraphes (``p.wp-block-paragraph``) et les listes
    (``ul.wp-block-list``) sont retenus : les ``div`` publicitaires et autres
    encarts intercalés sont ainsi ignorés d'office.
    """
    lines: list[str] = []
    for sibling in heading.find_next_siblings():
        if not isinstance(sibling, Tag):
            continue
        if sibling.name == "h2":
            break
        css_classes = sibling.get_attribute_list("class")
        if sibling.name == "p" and "wp-block-paragraph" in css_classes:
            text = clean_text(sibling.get_text())
            if text:
                lines.append(text)
        elif sibling.name == "ul" and "wp-block-list" in css_classes:
            for item in sibling.find_all("li"):
                text = clean_text(item.get_text())
                if text:
                    lines.append(f"- {text}")

    if not lines:
        raise UnexpectedStructureError(
            f"Réponse vide pour « {question} » sur {FAQ_URL} : "
            "aucun paragraphe ni liste entre ce titre et le suivant."
        )
    return "\n".join(lines)


def _extract_published_date(soup: BeautifulSoup) -> datetime | None:
    """Lit la date de publication ; ``None`` si le site ne l'expose plus."""
    meta = soup.select_one(PUBLISHED_META_SELECTOR)
    if meta is None:
        return None
    content = meta.get("content")
    if not isinstance(content, str):
        return None
    try:
        return datetime.fromisoformat(content)
    except ValueError as exc:
        raise UnexpectedStructureError(
            f"Date de publication illisible sur {FAQ_URL} : "
            f"« {content} » n'est pas une date ISO 8601."
        ) from exc
