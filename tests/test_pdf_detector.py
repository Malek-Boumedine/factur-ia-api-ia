"""Tests de la détection de la nature d'un PDF (natif vs scanné).

Les PDF factices sont générés en mémoire avec reportlab : un PDF natif avec du
texte, un PDF « scanné » sans couche texte (page vide). Le cas corrompu utilise
des octets bruts non valides.
"""

import io

import pytest
from reportlab.pdfgen import canvas
from src.extractions.pdf_detector import (
    PdfDetectionError,
    PdfType,
    detect_pdf_type,
)


def _native_pdf() -> bytes:
    """PDF natif : une page contenant du vrai texte extractible."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(
        100, 750, "Facture n 2026-042 - Total TTC 1234,56 EUR - Client Dupont"
    )
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _scanned_pdf() -> bytes:
    """PDF « scanné » : une page valide sans aucun objet texte."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    # Aucun drawString : la page ne contient pas de couche texte, comme un scan.
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_detect_native_pdf() -> None:
    assert detect_pdf_type(_native_pdf()) is PdfType.NATIVE


def test_detect_scanned_pdf() -> None:
    assert detect_pdf_type(_scanned_pdf()) is PdfType.SCANNED


def test_detect_corrupted_pdf_raises() -> None:
    with pytest.raises(PdfDetectionError):
        detect_pdf_type(b"%PDF-1.4 ceci n'est pas un vrai PDF")


def test_detect_empty_bytes_raises() -> None:
    with pytest.raises(PdfDetectionError):
        detect_pdf_type(b"")
