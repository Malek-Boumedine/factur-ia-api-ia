"""Tests de l'endpoint de réception ``POST /extractions``.

Périmètre : les **règles de validation du jeu de données d'entrée** — ce que le
service accepte de traiter et ce qu'il refuse avant même d'ouvrir le fichier
(authentification, type MIME, taille, forme de la requête). Ce qui se passe
*après* le ``202`` relève de ``test_pipeline_e2e.py``.

L'orchestrateur est remplacé pour toute la suite par une doublure qui enregistre
ses appels (fixture ``pipeline``, autouse) : les tests d'endpoint ne doivent pas
déclencher d'extraction réelle en tâche de fond, et cette doublure permet en
outre de vérifier qu'un fichier refusé n'atteint **jamais** le pipeline.
"""

import io

import pytest
from fastapi.testclient import TestClient
from src.core.config import settings
from src.extractions import router as router_module
from src.main import app

from tests.fixtures import documents

client = TestClient(app)

_VALID_HEADERS = {"X-OCR-Secret-Token": settings.SECRET_OCR_TOKEN}


class _PipelineSimule:
    """Doublure de l'orchestrateur : enregistre les extractions déclenchées."""

    def __init__(self) -> None:
        self.appels: list[tuple[bytes, int, str]] = []

    def __call__(self, content: bytes, id_document: int, content_type: str) -> None:
        self.appels.append((content, id_document, content_type))


@pytest.fixture(autouse=True)
def pipeline(monkeypatch: pytest.MonkeyPatch) -> _PipelineSimule:
    """Neutralise l'extraction en tâche de fond et capture ce qui est planifié."""
    simule = _PipelineSimule()
    monkeypatch.setattr(router_module, "run_extraction_pipeline", simule)
    return simule


def _fake_pdf() -> tuple[str, io.BytesIO, str]:
    """Petit fichier PDF factice (nom, contenu, type MIME)."""
    return ("facture.pdf", io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")


# --- Authentification -------------------------------------------------------


def test_receive_extraction_ok() -> None:
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": _fake_pdf()},
        data={"id_document": "42"},
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
        data={"id_document": "42"},
    )
    assert response.status_code == 403


def test_receive_extraction_missing_token() -> None:
    response = client.post(
        "/extractions",
        files={"file": _fake_pdf()},
        data={"id_document": "42"},
    )
    assert response.status_code == 422


def test_token_refuse_n_atteint_jamais_le_pipeline(pipeline: _PipelineSimule) -> None:
    """Un document non authentifié ne déclenche aucune extraction."""
    client.post(
        "/extractions",
        headers={"X-OCR-Secret-Token": "wrong-token"},  # pragma: allowlist secret
        files={"file": _fake_pdf()},
        data={"id_document": "42"},
    )

    assert pipeline.appels == []


# --- Règle : type de fichier ------------------------------------------------


@pytest.mark.parametrize(
    ("nom", "content_type"),
    [
        ("facture.pdf", "application/pdf"),
        ("facture.jpg", "image/jpeg"),
        ("facture.png", "image/png"),
    ],
)
def test_types_acceptes(pipeline: _PipelineSimule, nom: str, content_type: str) -> None:
    """Les trois formats du contrat sont acceptés et transmis au pipeline.

    Le type MIME est propagé tel quel : c'est lui qui décidera du routage entre
    extraction native et OCR.
    """
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": (nom, io.BytesIO(b"%PDF-1.4 contenu"), content_type)},
        data={"id_document": "42"},
    )

    assert response.status_code == 202
    assert len(pipeline.appels) == 1
    assert pipeline.appels[0][2] == content_type


@pytest.mark.parametrize(
    ("nom", "content_type"),
    [
        ("note.txt", "text/plain"),
        ("archive.zip", "application/zip"),
        ("anime.gif", "image/gif"),
        ("tableur.csv", "text/csv"),
    ],
)
def test_types_refuses(pipeline: _PipelineSimule, nom: str, content_type: str) -> None:
    """Tout format hors contrat est refusé en 400, sans atteindre le pipeline."""
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": (nom, io.BytesIO(b"contenu quelconque"), content_type)},
        data={"id_document": "42"},
    )

    assert response.status_code == 400
    assert pipeline.appels == []


# --- Règle : taille du fichier ----------------------------------------------


def test_fichier_trop_volumineux_refuse(
    monkeypatch: pytest.MonkeyPatch, pipeline: _PipelineSimule
) -> None:
    """Un fichier au-delà du plafond est refusé en 413, sans être traité.

    Le plafond est abaissé à 1 Mo pour le test : générer 10 Mo ne prouverait
    rien de plus et ralentirait la suite. Le refus doit intervenir *avant* toute
    planification d'extraction — c'est la protection contre l'épuisement des
    ressources.
    """
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)
    trop_gros = documents.octets_de_taille(1024 * 1024 + 1)

    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": ("enorme.pdf", trop_gros, "application/pdf")},
        data={"id_document": "42"},
    )

    assert response.status_code == 413
    assert "volumineux" in response.json()["detail"]
    assert pipeline.appels == []


def test_fichier_exactement_au_plafond_accepte(
    monkeypatch: pytest.MonkeyPatch, pipeline: _PipelineSimule
) -> None:
    """La borne est inclusive : un fichier pile à la limite passe.

    Vérifier la borne et pas seulement le dépassement : c'est là que se logent
    les erreurs de comparaison (``>`` vs ``>=``).
    """
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)
    pile_au_plafond = documents.octets_de_taille(1024 * 1024)

    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": ("limite.pdf", pile_au_plafond, "application/pdf")},
        data={"id_document": "42"},
    )

    assert response.status_code == 202
    assert len(pipeline.appels) == 1
    # Le fichier est transmis intégralement : la mesure de taille l'a rembobiné.
    assert len(pipeline.appels[0][0]) == 1024 * 1024


def test_fichier_vide_accepte_a_l_entree(pipeline: _PipelineSimule) -> None:
    """Un fichier de 0 octet passe la validation d'entrée.

    L'endpoint ne juge que le type et la taille ; un fichier vide est un
    document illisible, pas une requête malformée. Le verdict d'échec est émis
    par le pipeline (cf. ``test_pipeline_e2e``), ce qui laisse l'API data
    informée plutôt que de lui renvoyer une erreur HTTP sans suite.
    """
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": ("vide.pdf", io.BytesIO(b""), "application/pdf")},
        data={"id_document": "42"},
    )

    assert response.status_code == 202
    assert pipeline.appels == [(b"", 42, "application/pdf")]


# --- Règle : forme de la requête --------------------------------------------


def test_id_document_manquant_refuse(pipeline: _PipelineSimule) -> None:
    """Sans ``id_document``, l'extraction serait inexploitable côté API data."""
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": _fake_pdf()},
    )

    assert response.status_code == 422
    assert pipeline.appels == []


@pytest.mark.parametrize("valeur", ["abc", "", "12.5"])
def test_id_document_non_entier_refuse(pipeline: _PipelineSimule, valeur: str) -> None:
    """``id_document`` doit être un entier : le contrat en dépend."""
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": _fake_pdf()},
        data={"id_document": valeur},
    )

    assert response.status_code == 422
    assert pipeline.appels == []


def test_fichier_manquant_refuse(pipeline: _PipelineSimule) -> None:
    """Une requête sans fichier est rejetée avant tout traitement."""
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        data={"id_document": "42"},
    )

    assert response.status_code == 422
    assert pipeline.appels == []


# --- Planification de l'extraction ------------------------------------------


def test_receive_extraction_schedules_pipeline(pipeline: _PipelineSimule) -> None:
    """Le 202 déclenche l'orchestrateur en tâche de fond avec les bons arguments.

    Les octets sont lus dans le handler (avant le ``202``) : l'``UploadFile``
    peut être fermé quand la tâche s'exécute, ce sont donc des octets immuables
    qui sont transmis.
    """
    response = client.post(
        "/extractions",
        headers=_VALID_HEADERS,
        files={"file": _fake_pdf()},
        data={"id_document": "42"},
    )

    assert response.status_code == 202
    # TestClient exécute la tâche de fond après la réponse.
    assert pipeline.appels == [(b"%PDF-1.4 fake content", 42, "application/pdf")]
