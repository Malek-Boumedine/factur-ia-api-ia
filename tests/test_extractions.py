"""Tests de l'endpoint de réception POST /extractions."""

import io

from fastapi.testclient import TestClient
from src.core.config import settings
from src.main import app

client = TestClient(app)

_VALID_HEADERS = {"X-OCR-Secret-Token": settings.SECRET_OCR_TOKEN}


def _fake_pdf() -> tuple[str, io.BytesIO, str]:
    """Petit fichier PDF factice (nom, contenu, type MIME)."""
    return ("facture.pdf", io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")


def test_receive_extraction_ok() -> None:
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": _fake_pdf()},
        data={"id_document": 42},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["id_document"] == 42
    assert "message" in body


def test_receive_extraction_invalid_token() -> None:
    response = client.post(
        "/extractions",
        headers={"X-OCR-Secret-Token": "wrong-token"},  # pragma: allowlist secret
        files={"file": _fake_pdf()},
        data={"id_document": 42},
    )
    assert response.status_code == 403


def test_receive_extraction_unsupported_type() -> None:
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": ("note.txt", io.BytesIO(b"juste du texte"), "text/plain")},
        data={"id_document": 42},
    )
    assert response.status_code == 400


def test_receive_extraction_missing_token() -> None:
    response = client.post(
        "/extractions",
        files={"file": _fake_pdf()},
        data={"id_document": 42},
    )
    assert response.status_code == 422
