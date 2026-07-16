"""Tests de l'extraction du texte par OCR (images et PDF scannés).

EasyOCR est systématiquement mocké : aucun téléchargement de modèle ni vrai OCR
en CI. On teste la *logique* (conversion PDF→image via PyMuPDF, appel de l'OCR
page par page, concaténation, gestion d'erreur, cache du Reader), pas la qualité
de reconnaissance d'EasyOCR lui-même.

La conversion PyMuPDF est exercée pour de vrai : les fixtures PDF sont générées
avec reportlab, seul l'OCR est remplacé par un faux Reader.
"""

import io

import pytest
from reportlab.pdfgen import canvas
from src.extractions import ocr_extractor
from src.extractions.ocr_extractor import OcrExtractionError, extract_ocr_text


class _FakeReader:
    """Faux Reader EasyOCR : renvoie des fragments fixes et compte les appels.

    ``readtext(detail=0)`` renvoie ``pages[i]`` (liste de fragments) au i-ème
    appel, permettant de simuler un OCR distinct par page.
    """

    def __init__(self, *pages: list[str]) -> None:
        self._pages = list(pages)
        self.calls = 0

    def readtext(self, image: bytes, detail: int = 1) -> list[str]:
        fragments = self._pages[self.calls] if self.calls < len(self._pages) else []
        self.calls += 1
        return fragments


def _pdf_with_pages(n: int) -> bytes:
    """PDF de ``n`` pages (contenu quelconque : l'OCR est mocké)."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for i in range(n):
        pdf.drawString(100, 750, f"page {i}")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _reset_reader_cache() -> None:
    """Réinitialise le cache module-level du Reader entre les tests."""
    ocr_extractor._reader = None


def test_ocr_image_returns_joined_text(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _FakeReader(["Facture 2026-042", "Total TTC 1234,56 EUR"])
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)

    result = extract_ocr_text(b"fake-image-bytes", is_pdf=False)

    assert "Facture 2026-042" in result
    assert "Total TTC 1234,56 EUR" in result
    assert reader.calls == 1


def test_ocr_scanned_pdf_converts_each_page(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deux pages, un OCR distinct par page : vérifie la vraie conversion PyMuPDF
    # (2 pages → 2 images → 2 appels OCR) et la concaténation dans l'ordre.
    reader = _FakeReader(["PREMIERE PAGE"], ["DEUXIEME PAGE"])
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)

    result = extract_ocr_text(_pdf_with_pages(2), is_pdf=True)

    assert reader.calls == 2
    assert "PREMIERE PAGE" in result
    assert "DEUXIEME PAGE" in result
    assert result.index("PREMIERE PAGE") < result.index("DEUXIEME PAGE")


def test_ocr_corrupted_pdf_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _FakeReader(["ne devrait pas etre appele"])
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)

    with pytest.raises(OcrExtractionError):
        extract_ocr_text(b"%PDF-1.4 ceci n'est pas un vrai PDF", is_pdf=True)


def test_ocr_reader_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingReader:
        def readtext(self, image: bytes, detail: int = 1) -> list[str]:
            raise ValueError("image corrompue")

    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: _FailingReader())

    with pytest.raises(OcrExtractionError):
        extract_ocr_text(b"fake-image-bytes", is_pdf=False)


def test_ocr_empty_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _FakeReader([])  # aucun fragment reconnu
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)

    with pytest.raises(OcrExtractionError):
        extract_ocr_text(b"fake-image-bytes", is_pdf=False)


def test_ocr_pdf_over_page_cap_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un PDF scanné au-delà du plafond de pages est rejeté AVANT tout rendu/OCR
    (borne temps + mémoire), en OcrExtractionError → payload d'échec en aval."""
    reader = _FakeReader(["ne devrait pas etre appele"])
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)
    monkeypatch.setattr(ocr_extractor, "_MAX_OCR_PAGES", 2)

    with pytest.raises(OcrExtractionError):
        extract_ocr_text(_pdf_with_pages(3), is_pdf=True)

    assert reader.calls == 0  # rejeté avant tout appel OCR


def test_ocr_pdf_at_page_cap_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un PDF exactement au plafond passe normalement (la borne est inclusive)."""
    reader = _FakeReader(["PAGE 1"], ["PAGE 2"])
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: reader)
    monkeypatch.setattr(ocr_extractor, "_MAX_OCR_PAGES", 2)

    result = extract_ocr_text(_pdf_with_pages(2), is_pdf=True)

    assert reader.calls == 2
    assert "PAGE 1" in result
    assert "PAGE 2" in result


def test_reader_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Le Reader (coûteux) ne doit être construit qu'une seule fois puis réutilisé.
    calls = {"count": 0}

    class _FakeEasyocr:
        @staticmethod
        def Reader(languages: list[str], gpu: bool) -> _FakeReader:
            calls["count"] += 1
            return _FakeReader()

    monkeypatch.setitem(__import__("sys").modules, "easyocr", _FakeEasyocr)

    first = ocr_extractor._get_reader()
    second = ocr_extractor._get_reader()

    assert first is second
    assert calls["count"] == 1
