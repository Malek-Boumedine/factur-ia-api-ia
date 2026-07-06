"""Tests de l'extraction du texte des PDF natifs.

Fixtures générées en mémoire avec reportlab : PDF natifs avec texte connu,
mono et multi-pages, plus un cas de page vide au milieu et un PDF corrompu.
"""

import io

import pytest
from reportlab.pdfgen import canvas
from src.extractions.pdf_extractor import PdfExtractionError, extract_native_pdf_text


def _pdf_with_pages(*pages: str | None) -> bytes:
    """PDF natif dont chaque argument est le texte d'une page.

    ``None`` produit une page sans texte (comme une page image au milieu).
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        if text is not None:
            pdf.drawString(100, 750, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


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
        extract_native_pdf_text(b"%PDF-1.4 ceci n'est pas un vrai PDF")


def test_extract_pdf_without_text_raises() -> None:
    content = _pdf_with_pages(None, None)
    with pytest.raises(PdfExtractionError):
        extract_native_pdf_text(content)
