"""Extraction du texte brut d'un PDF natif via ``pdfplumber``.

Étape qui suit la détection : un PDF classé ``NATIVE`` par ``detect_pdf_type``
contient du texte extractible directement. On récupère ce texte brut, concaténé
dans l'ordre des pages, pour l'envoyer ensuite au LLM (structuration — étape
ultérieure).

Ce module ne fait *que* l'extraction du texte des PDF natifs : ni OCR (PDF
scannés / images), ni structuration LLM. Le nettoyage est volontairement léger
pour préserver la fidélité du contenu (alignement des colonnes, montants), dont
le LLM a besoin.
"""

import io
import re

import pdfplumber


class PdfExtractionError(Exception):
    """PDF illisible, corrompu, ou sans texte extractible.

    Relevée quand ``pdfplumber`` ne parvient pas à ouvrir/parcourir le document,
    ou quand un PDF supposé natif ne livre finalement aucun texte. L'orchestrateur
    du pipeline attrape cette exception pour produire un résultat d'échec
    (``score_confiance = 0``) côté API data.
    """


def _clean_page_text(text: str) -> str:
    """Nettoyage minimal du texte d'une page.

    Retire les espaces de fin de ligne (parfois ajoutés par pdfplumber) et les
    lignes vides en tête/fin. Les espaces internes et l'alignement des colonnes
    sont préservés : le LLM a besoin de la structure la plus fidèle possible.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def extract_native_pdf_text(content: bytes) -> str:
    """Extrait le texte brut d'un PDF natif, concaténé dans l'ordre des pages.

    Les pages sont jointes par une ligne vide (``\\n\\n``), séparateur naturel
    sans marqueur artificiel. Les pages sans texte (une éventuelle image au
    milieu d'un PDF natif) sont ignorées, pas traitées comme une erreur.

    Args:
        content: contenu binaire du fichier PDF (détecté ``NATIVE`` en amont).

    Returns:
        Le texte brut concaténé, nettoyé a minima.

    Raises:
        PdfExtractionError: le contenu n'est pas un PDF lisible, ou aucun texte
            n'a pu être extrait (PDF supposé natif mais sans couche texte).
    """
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages_text = [
                cleaned
                for page in pdf.pages
                if (cleaned := _clean_page_text(page.extract_text() or ""))
            ]
    except Exception as exc:  # pdfminer lève des exceptions variées et peu typées
        raise PdfExtractionError(
            "PDF illisible ou corrompu : extraction du texte impossible."
        ) from exc

    # Réduit les gros trous (3+ sauts de ligne) sans écraser la structure utile.
    text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(pages_text)).strip()

    if not text:
        raise PdfExtractionError(
            "PDF natif sans texte extractible : extraction impossible."
        )

    return text
