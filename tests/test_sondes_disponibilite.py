"""Tests des sondes de disponibilité (``GET /health`` et ``GET /ready``).

Sans réseau : le ``TestClient`` parle à l'application par un transport ASGI en
mémoire, et la seule dépendance sondée (les poids EasyOCR) est un répertoire sur
disque, simulé ici par un ``tmp_path``. La garde réseau autouse de
``conftest.py`` reste donc satisfaite sans aménagement — et c'est justement ce
qu'on vérifie pour ``/ready``, qui ne doit toucher ni Groq ni l'API data.

Le répertoire modèle est piloté par la variable d'environnement
``EASYOCR_MODULE_PATH``, qu'``ocr_model_available`` consulte comme le fait
EasyOCR : les tests n'ont donc rien à monkeypatcher dans le code de production.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.extractions import llm_client
from src.main import app

client = TestClient(app)


@pytest.fixture
def repertoire_modele(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isole le répertoire de poids EasyOCR dans un ``tmp_path`` vide.

    Renvoie le répertoire ``model/`` (non créé) : à chaque test de le peupler ou
    de le laisser absent selon le cas simulé.
    """
    monkeypatch.setenv("EASYOCR_MODULE_PATH", str(tmp_path))
    return tmp_path / "model"


def _installer_poids(repertoire: Path) -> None:
    """Simule des poids OCR installés (contenu sans importance : seule la
    présence d'un fichier ``.pth`` est testée)."""
    repertoire.mkdir(parents=True)
    (repertoire / "craft_mlt_25k.pth").write_bytes(b"poids factices")


def test_health_repond_sans_aucune_dependance(repertoire_modele: Path) -> None:
    """/health répond 200 alors même que le répertoire modèle est absent.

    C'est la propriété essentielle de la sonde de liveness : son échec
    redémarrerait le conteneur, elle ne doit donc dépendre de rien — pas même
    de ce que vérifie /ready.
    """
    assert not repertoire_modele.exists()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ne_divulgue_rien() -> None:
    """Le corps se limite au statut : ni version, ni nom d'application, ni
    environnement d'exécution ne fuitent par cette route publique."""
    body = client.get("/health").json()

    assert set(body.keys()) == {"status"}


def test_ready_200_quand_les_poids_sont_installes(
    repertoire_modele: Path,
) -> None:
    """/ready répond 200 dès qu'au moins un fichier de poids est présent."""
    _installer_poids(repertoire_modele)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize("creer_repertoire", [False, True], ids=["absent", "vide"])
def test_ready_503_quand_les_poids_manquent(
    repertoire_modele: Path, creer_repertoire: bool
) -> None:
    """Sans poids OCR, /ready répond 503 avec un corps minimal : l'instance est
    retirée du trafic, et aucun détail exploitable ne fuit dans la réponse.

    Les deux formes de l'absence sont couvertes : répertoire jamais créé
    (instance fraîche, système de fichiers éphémère) et répertoire présent mais
    vide (image construite sans les poids).
    """
    if creer_repertoire:
        repertoire_modele.mkdir(parents=True)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "service non prêt"}


def test_ready_ne_construit_aucun_client_groq(
    repertoire_modele: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La sonde ne touche jamais au service payant.

    Le client Groq est mis en cache au niveau module à sa première utilisation :
    s'il vaut toujours ``None`` après un appel à /ready, c'est qu'aucun appel LLM
    n'a été tenté. La garde réseau de ``conftest.py`` couvre déjà le cas d'une
    connexion réelle ; ce test couvre l'intention, avant même la connexion.

    Le cache est remis à zéro via ``monkeypatch`` (donc restauré ensuite) : les
    tests du client LLM le peuplent, et sans cette base propre l'assertion
    dépendrait de l'ordre d'exécution de la suite.
    """
    monkeypatch.setattr(llm_client, "_client", None)
    _installer_poids(repertoire_modele)

    assert client.get("/ready").status_code == 200

    assert llm_client._client is None


def test_sondes_hors_contrat_openapi() -> None:
    """Routes d'infrastructure : absentes du schéma OpenAPI, qui ne décrit que
    le contrat métier exposé à l'API data."""
    paths = app.openapi()["paths"]

    assert "/health" not in paths
    assert "/ready" not in paths
