"""Détection de la nature d'un PDF : natif (texte extractible) vs scanné.

Première étape du pipeline d'extraction. Un PDF natif (généré par un logiciel
bureautique) contient du texte extractible directement via ``pdfplumber`` ; un
PDF scanné (photo/scan d'une facture papier) n'a pas de couche texte et devra
passer par l'OCR (EasyOCR).

Ce module ne fait *que* décider quelle voie prendre. L'extraction elle-même
(pdfplumber ou OCR) est réalisée par les étapes suivantes du pipeline.

Note : les images (JPEG/PNG) ne sont pas des PDF et ne passent jamais par ce
détecteur ; elles sont routées directement vers l'OCR par l'orchestrateur.
"""

import io
from enum import StrEnum
from typing import Any

import pdfplumber

# Seuil minimal de caractères « réels » (alphanumériques) sur l'ensemble du
# document pour le considérer comme natif. Un scan pur renvoie 0 caractère ;
# un vrai document natif dépasse très largement ce seuil bas. Volontairement
# permissif : en cas de doute on privilégie « natif » sauf absence quasi totale
# de texte. Détail d'implémentation (pas dans Settings) ; pourra être remonté en
# configuration si un ajustement sans redéploiement devient nécessaire.
_MIN_TEXT_CHARS = 20


class PdfType(StrEnum):
    """Nature d'un PDF vis-à-vis de l'extraction de texte."""

    NATIVE = "native"  # texte extractible → pdfplumber
    SCANNED = "scanned"  # pas de texte exploitable → OCR (EasyOCR)


class PdfDetectionError(Exception):
    """PDF illisible ou corrompu : la détection est impossible.

    Relevée quand ``pdfplumber`` ne parvient pas à ouvrir ou parcourir le
    document. L'orchestrateur du pipeline attrape cette exception pour produire
    un résultat d'échec (``score_confiance = 0``) côté API data.
    """


def _count_real_text_chars(pdf: Any) -> int:  # pdfplumber n'expose pas de stubs
    """Compte les caractères alphanumériques extraits sur toutes les pages.

    On ignore espaces, ponctuation et artefacts de mise en page pour ne mesurer
    que du contenu textuel réel.
    """
    total = 0
    for page in pdf.pages:
        text = page.extract_text() or ""
        total += sum(1 for char in text if char.isalnum())
    return total


def detect_pdf_type(content: bytes) -> PdfType:
    """Détermine si un PDF est natif (texte) ou scanné (image → OCR).

    Ouvre le document, extrait le texte de chaque page et compte les caractères
    alphanumériques réels. Au-dessus du seuil ``_MIN_TEXT_CHARS`` le PDF est
    considéré natif, sinon scanné.

    Args:
        content: contenu binaire du fichier PDF.

    Returns:
        ``PdfType.NATIVE`` si du texte exploitable est présent, sinon
        ``PdfType.SCANNED``.

    Raises:
        PdfDetectionError: le contenu n'est pas un PDF lisible (corrompu,
            tronqué, ou format inattendu).

    Limites connues :
        - PDF hybride (image scannée + fine couche texte / tampon OCR) : classé
          natif dès qu'il dépasse le seuil, même si l'essentiel est en image.
        - PDF natif quasi vide : risque de faux « scanné » ; le pire cas envoie
          un PDF natif vers l'OCR (dégradé, non bloquant).
        - Seuil global au document (non par page) : adapté à des factures de
          quelques pages.
    """
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            real_chars = _count_real_text_chars(pdf)
    except PdfDetectionError:
        raise
    except Exception as exc:  # pdfminer lève des exceptions variées et peu typées
        raise PdfDetectionError(
            "PDF illisible ou corrompu : détection de la nature impossible."
        ) from exc

    return PdfType.NATIVE if real_chars >= _MIN_TEXT_CHARS else PdfType.SCANNED
