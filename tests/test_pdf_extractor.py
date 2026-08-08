"""Tests de l'extraction du texte des PDF natifs.

Documents issus de la fabrique partagée (``tests/fixtures/documents.py``),
générés en mémoire : PDF natifs avec texte connu, mono et multi-pages, plus un
cas de page vide au milieu et un PDF corrompu.
"""

import pytest
from src.extractions.pdf_extractor import PdfExtractionError, extract_native_pdf_text

from tests.fixtures import documents
from tests.fixtures.documents import pdf_with_pages as _pdf_with_pages


def test_extract_single_page_text() -> None:
    content = _pdf_with_pages("Facture 2026-042 Total TTC 1234,56 EUR")
    result = extract_native_pdf_text(content)
    assert "Facture 2026-042" in result
    assert "1234,56 EUR" in result


def test_extract_multi_page_preserves_order() -> None:
    content = _pdf_with_pages("PREMIERE PAGE contenu un", "DEUXIEME PAGE contenu deux")
    result = extract_native_pdf_text(content)
    assert "PREMIERE PAGE" in result
    assert "DEUXIEME PAGE" in result
    assert result.index("PREMIERE PAGE") < result.index("DEUXIEME PAGE")


def test_extract_skips_empty_middle_page() -> None:
    content = _pdf_with_pages("PAGE UNE", None, "PAGE TROIS")
    result = extract_native_pdf_text(content)
    assert "PAGE UNE" in result
    assert "PAGE TROIS" in result


def test_extract_corrupted_pdf_raises() -> None:
    with pytest.raises(PdfExtractionError):
        extract_native_pdf_text(documents.PDF_CORROMPU)


def test_extract_pdf_without_text_raises() -> None:
    content = _pdf_with_pages(None, None)
    with pytest.raises(PdfExtractionError):
        extract_native_pdf_text(content)
