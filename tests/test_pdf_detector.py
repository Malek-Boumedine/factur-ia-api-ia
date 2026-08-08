"""Tests de la détection de la nature d'un PDF (natif vs scanné).

Les documents viennent de la fabrique partagée (``tests/fixtures/documents.py``) :
générés en mémoire, jamais versionnés. Un PDF natif porte du texte extractible,
un PDF « scanné » est une page valide sans couche texte. Le cas corrompu utilise
des octets bruts non valides.
"""

import pytest
from src.extractions.pdf_detector import (
    PdfDetectionError,
    PdfType,
    detect_pdf_type,
)

from tests.fixtures import documents


def test_detect_native_pdf() -> None:
    assert detect_pdf_type(documents.facture_native_pdf()) is PdfType.NATIVE


def test_detect_scanned_pdf() -> None:
    assert detect_pdf_type(documents.pdf_scanne()) is PdfType.SCANNED


def test_detect_corrupted_pdf_raises() -> None:
    with pytest.raises(PdfDetectionError):
        detect_pdf_type(documents.PDF_CORROMPU)


def test_detect_empty_bytes_raises() -> None:
    with pytest.raises(PdfDetectionError):
        detect_pdf_type(b"")
