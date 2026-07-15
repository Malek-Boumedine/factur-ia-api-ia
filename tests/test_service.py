"""Tests de l'orchestrateur du pipeline (``run_extraction_pipeline``).

Les briques lourdes (OCR EasyOCR, LLM Groq, ouverture PDF ``pdfplumber``) sont
mockées : aucun appel réseau, aucun vrai OCR. L'envoi au callback OCR
(``send_callback``) est mocké par une fixture autouse qui capture les payloads
envoyés — aucun POST réel. On teste le *câblage* — routage des trois chemins
(image, PDF natif, PDF scanné), enchaînement des étapes, forme du payload final,
chemin d'échec, envoi au callback — pas le comportement des briques elles-mêmes.

``compute_confidence`` et ``validate_extraction`` (purs, déterministes, sans
réseau) restent réels : le payload produit est donc validé de bout en bout contre
le contrat ``OcrWebhookPayload``.
"""

from decimal import Decimal
from typing import Any

import pytest
from src.callback.client import CallbackError
from src.callback.schemas import OcrWebhookPayload
from src.extractions import service
from src.extractions.ocr_extractor import OcrExtractionError
from src.extractions.pdf_detector import PdfType
from src.extractions.prompts import TypeDocument
from src.extractions.service import run_extraction_pipeline

_PDF_MIME = "application/pdf"
_PNG_MIME = "image/png"


def _facture() -> dict[str, Any]:
    """Sous-ensemble « données extraites » cohérent (HT + TVA = TTC), en Decimal."""
    return {
        "siret_emetteur": "12345678900011",
        "siret_destinataire": "98765432100022",
        "numero_facture": "FA-2026-042",
        "date_emission": "2026-07-06",
        "total_ht": Decimal("1000.00"),
        "total_tva": Decimal("200.00"),
        "total_ttc": Decimal("1200.00"),
        "iban": "FR7630006000011234567890189",
        "lignes": [
            {
                "designation": "Prestation de conseil",
                "quantite": Decimal("2"),
                "prix_unitaire_ht": Decimal("500.00"),
                "taux_tva": Decimal("20"),
            }
        ],
    }


@pytest.fixture(autouse=True)
def sent_payloads(monkeypatch: pytest.MonkeyPatch) -> list[OcrWebhookPayload]:
    """Mocke l'envoi au callback OCR : aucun POST réel, capture les payloads.

    Autouse : sans elle, chaque test du pipeline tenterait un vrai POST vers
    ``settings.ocr_callback_url``. Les tests qui veulent vérifier l'envoi
    demandent la fixture et inspectent la liste des payloads capturés.
    """
    sent: list[OcrWebhookPayload] = []
    monkeypatch.setattr(service, "send_callback", sent.append)
    return sent


@pytest.fixture
def mock_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocke la structuration LLM : renvoie une facture cohérente + type suggéré."""

    def _fake_structure(raw_text: str) -> dict[str, Any]:
        return {"type_document": TypeDocument.FACTURE, "facture": _facture()}

    monkeypatch.setattr(service, "structure_invoice", _fake_structure)


def test_image_path_goes_straight_to_ocr(
    monkeypatch: pytest.MonkeyPatch, mock_structure: None
) -> None:
    """Une image (PNG) va directement à l'OCR (is_pdf=False), sans détecteur PDF."""
    ocr_calls: list[dict[str, Any]] = []

    def _fake_ocr(content: bytes, *, is_pdf: bool) -> str:
        ocr_calls.append({"content": content, "is_pdf": is_pdf})
        return "texte OCR image"

    def _fail_detect(content: bytes) -> PdfType:
        raise AssertionError("le détecteur PDF ne doit pas être appelé pour une image")

    monkeypatch.setattr(service, "extract_ocr_text", _fake_ocr)
    monkeypatch.setattr(service, "detect_pdf_type", _fail_detect)

    payload = run_extraction_pipeline(b"image-bytes", 42, _PNG_MIME)

    assert len(ocr_calls) == 1
    assert ocr_calls[0]["is_pdf"] is False
    assert ocr_calls[0]["content"] == b"image-bytes"
    assert payload.id_document == 42


def test_native_pdf_path(monkeypatch: pytest.MonkeyPatch, mock_structure: None) -> None:
    """Un PDF natif passe par le détecteur puis l'extraction native, pas l'OCR."""
    native_calls: list[bytes] = []

    def _fake_native(content: bytes) -> str:
        native_calls.append(content)
        return "texte PDF natif"

    def _fail_ocr(content: bytes, *, is_pdf: bool) -> str:
        raise AssertionError("l'OCR ne doit pas être appelé pour un PDF natif")

    monkeypatch.setattr(service, "detect_pdf_type", lambda content: PdfType.NATIVE)
    monkeypatch.setattr(service, "extract_native_pdf_text", _fake_native)
    monkeypatch.setattr(service, "extract_ocr_text", _fail_ocr)

    payload = run_extraction_pipeline(b"pdf-bytes", 7, _PDF_MIME)

    assert native_calls == [b"pdf-bytes"]
    assert payload.id_document == 7


def test_scanned_pdf_path(
    monkeypatch: pytest.MonkeyPatch, mock_structure: None
) -> None:
    """Un PDF scanné passe par le détecteur puis l'OCR avec is_pdf=True."""
    ocr_calls: list[dict[str, Any]] = []

    def _fake_ocr(content: bytes, *, is_pdf: bool) -> str:
        ocr_calls.append({"content": content, "is_pdf": is_pdf})
        return "texte OCR PDF scanné"

    def _fail_native(content: bytes) -> str:
        raise AssertionError("l'extraction native ne doit pas être appelée sur un scan")

    monkeypatch.setattr(service, "detect_pdf_type", lambda content: PdfType.SCANNED)
    monkeypatch.setattr(service, "extract_ocr_text", _fake_ocr)
    monkeypatch.setattr(service, "extract_native_pdf_text", _fail_native)

    payload = run_extraction_pipeline(b"scan-bytes", 9, _PDF_MIME)

    assert len(ocr_calls) == 1
    assert ocr_calls[0]["is_pdf"] is True
    assert ocr_calls[0]["content"] == b"scan-bytes"
    assert payload.id_document == 9


def test_final_payload_is_correct(
    monkeypatch: pytest.MonkeyPatch, mock_structure: None
) -> None:
    """Le payload final reflète l'extraction : score > 0 et champs facture remontés."""
    monkeypatch.setattr(service, "extract_ocr_text", lambda content, *, is_pdf: "txt")

    payload = run_extraction_pipeline(b"image", 123, _PNG_MIME)

    assert payload.id_document == 123
    assert payload.score_confiance > 0  # extraction réussie → jamais le sentinelle 0
    assert payload.total_ht == Decimal("1000.00")
    assert payload.total_tva == Decimal("200.00")
    assert payload.total_ttc == Decimal("1200.00")
    assert payload.numero_facture == "FA-2026-042"
    assert len(payload.lignes) == 1
    assert payload.lignes[0].designation == "Prestation de conseil"


def test_extraction_failure_yields_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Une brique en échec (OCR illisible) → payload d'échec (score_confiance = 0)."""

    def _raise(content: bytes, *, is_pdf: bool) -> str:
        raise OcrExtractionError("image illisible")

    monkeypatch.setattr(service, "extract_ocr_text", _raise)

    payload = run_extraction_pipeline(b"corrompu", 55, _PNG_MIME)

    assert payload.id_document == 55
    assert payload.score_confiance == Decimal("0")  # marqueur unique d'échec
    assert payload.total_ht == Decimal("0")
    assert payload.lignes == []


def test_success_payload_is_sent_to_callback(
    monkeypatch: pytest.MonkeyPatch,
    mock_structure: None,
    sent_payloads: list[OcrWebhookPayload],
) -> None:
    """Le payload de succès est envoyé au callback (le même que celui retourné)."""
    monkeypatch.setattr(service, "extract_ocr_text", lambda content, *, is_pdf: "txt")

    payload = run_extraction_pipeline(b"image", 42, _PNG_MIME)

    assert sent_payloads == [payload]
    assert sent_payloads[0].score_confiance > 0


def test_failure_payload_is_also_sent_to_callback(
    monkeypatch: pytest.MonkeyPatch,
    sent_payloads: list[OcrWebhookPayload],
) -> None:
    """Le payload d'échec part AUSSI au callback : c'est ainsi que l'API data
    apprend l'échec et passe le document en « erreur »."""

    def _raise(content: bytes, *, is_pdf: bool) -> str:
        raise OcrExtractionError("image illisible")

    monkeypatch.setattr(service, "extract_ocr_text", _raise)

    payload = run_extraction_pipeline(b"corrompu", 55, _PNG_MIME)

    assert sent_payloads == [payload]
    assert sent_payloads[0].id_document == 55
    assert sent_payloads[0].score_confiance == Decimal("0")


def test_callback_error_does_not_crash_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    mock_structure: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Un échec définitif du callback est journalisé et avalé : la tâche de fond
    ne plante pas, le pipeline retourne quand même le payload."""
    monkeypatch.setattr(service, "extract_ocr_text", lambda content, *, is_pdf: "txt")

    def _fail_send(payload: OcrWebhookPayload) -> None:
        raise CallbackError("callback définitivement échoué")

    monkeypatch.setattr(service, "send_callback", _fail_send)

    with caplog.at_level("ERROR"):
        payload = run_extraction_pipeline(b"image", 42, _PNG_MIME)

    assert payload.id_document == 42  # pas d'exception propagée
    assert "callback" in caplog.text
