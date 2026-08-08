"""Tests du monitoring de la qualité d'extraction (``core/monitoring.py``).

Aucun serveur MLflow n'est nécessaire : le traçage écrit dans une base SQLite
temporaire (``tmp_path``), relue ensuite avec le client MLflow. C'est la
propriété qui rend ces tests exécutables en CI sans service tiers.

Trois familles de tests : le calcul des taux (fonctions pures), le contenu
effectivement écrit dans le run (métriques, tags, absence de donnée sensible),
et les garanties de non-nuisance (désactivé par défaut, ne lève jamais).
"""

from decimal import Decimal
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from src.callback.schemas import LigneOcr, OcrWebhookPayload
from src.core import monitoring
from src.core.config import settings
from src.core.monitoring import (
    ModeExtraction,
    taux_champs_presents,
    taux_champs_reconnus,
    track_extraction_quality,
)

_EXPERIENCE_TEST = "test-qualite-extraction"

# Valeurs sensibles plantées dans le payload de test : aucune ne doit se
# retrouver dans le run tracé (cf. ``test_aucune_donnee_sensible_n_est_tracee``).
_SIRET_EMETTEUR = "73282932000074"
_SIRET_DESTINATAIRE = "55208131766522"
_IBAN = "FR7630006000011234567890189"
_NUMERO_FACTURE = "FA-2026-042"
_DESIGNATION = "Prestation de conseil"
_TOTAL_HT = Decimal("1234.56")
_TOTAL_TVA = Decimal("246.91")
_TOTAL_TTC = Decimal("1481.47")

_VALEURS_SENSIBLES = (
    _SIRET_EMETTEUR,
    _SIRET_DESTINATAIRE,
    _IBAN,
    _NUMERO_FACTURE,
    _DESIGNATION,
    str(_TOTAL_HT),
    str(_TOTAL_TVA),
    str(_TOTAL_TTC),
    "2026-07-06",
)


def _par_champ(**surcharges: Decimal) -> dict[str, Decimal]:
    """Confiances par champ, toutes bonnes par défaut, surchargeables."""
    scores = dict.fromkeys(monitoring.TRACKED_FIELDS, Decimal("1.0000"))
    scores.update(surcharges)
    return scores


def _payload_succes(**surcharges: object) -> OcrWebhookPayload:
    """Payload d'extraction réussie, garni de valeurs sensibles réalistes."""
    champs: dict[str, object] = {
        "id_document": 42,
        "score_confiance": Decimal("0.9000"),
        "siret_emetteur": _SIRET_EMETTEUR,
        "siret_destinataire": _SIRET_DESTINATAIRE,
        "numero_facture": _NUMERO_FACTURE,
        "date_emission": "2026-07-06",
        "date_echeance": "2026-08-05",
        "total_ht": _TOTAL_HT,
        "total_tva": _TOTAL_TVA,
        "total_ttc": _TOTAL_TTC,
        "iban": _IBAN,
        "lignes": [
            LigneOcr(
                designation=_DESIGNATION,
                quantite=Decimal("2"),
                prix_unitaire_ht=Decimal("617.28"),
                taux_tva=Decimal("20"),
            )
        ],
        "type_document": "facture",
        "par_champ": _par_champ(),
    }
    champs.update(surcharges)
    return OcrWebhookPayload(**champs)  # type: ignore[arg-type]


def _payload_echec() -> OcrWebhookPayload:
    """Payload d'échec canonique : marqueur ``score_confiance = 0``, pas d'options."""
    return OcrWebhookPayload(
        id_document=55,
        score_confiance=Decimal("0"),
        total_ht=Decimal("0"),
        total_tva=Decimal("0"),
        total_ttc=Decimal("0"),
        lignes=[],
    )


@pytest.fixture
def store_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Active le traçage sur une base SQLite temporaire, isolée par test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(settings, "MLFLOW_EXPERIMENT_NAME", _EXPERIENCE_TEST)
    return uri


def _dernier_run(store: str) -> mlflow.entities.Run:
    """Relit l'unique run écrit dans le store temporaire."""
    client = MlflowClient(tracking_uri=store)
    experience = client.get_experiment_by_name(_EXPERIENCE_TEST)
    assert experience is not None, "l'expérience n'a pas été créée"
    runs = client.search_runs([experience.experiment_id])
    assert len(runs) == 1, f"un seul run attendu, {len(runs)} trouvé(s)"
    return runs[0]


# --- Calcul des taux (fonctions pures) -------------------------------------


def test_taux_sur_extraction_parfaite() -> None:
    """Tous les champs à 1 : les deux taux valent 1.0."""
    par_champ = _par_champ()

    assert taux_champs_reconnus(par_champ) == 1.0
    assert taux_champs_presents(par_champ) == 1.0


def test_taux_distingue_champ_absent_et_champ_douteux() -> None:
    """Un champ absent (0) et un champ douteux (0.4) comptent différemment.

    C'est tout l'intérêt des deux taux : le douteux est *présent* mais pas
    *reconnu*, l'absent n'est ni l'un ni l'autre. Deux problèmes distincts.
    """
    par_champ = _par_champ(
        iban=Decimal("0"),  # absent
        siret_emetteur=Decimal("0.4000"),  # présent mais invalidé
    )

    assert taux_champs_reconnus(par_champ) == 0.8  # 8 champs sur 10 ≥ 0.7
    assert taux_champs_presents(par_champ) == 0.9  # 9 champs sur 10 > 0


def test_seuil_de_reconnaissance_inclusif() -> None:
    """Un champ exactement au seuil (0.7) est compté comme reconnu."""
    assert taux_champs_reconnus(_par_champ(iban=Decimal("0.7000"))) == 1.0
    assert taux_champs_reconnus(_par_champ(iban=Decimal("0.6999"))) == 0.9


def test_taux_nuls_sans_confiance_par_champ() -> None:
    """``par_champ`` absent (payload d'échec) → taux à 0, série continue."""
    assert taux_champs_reconnus(None) == 0.0
    assert taux_champs_presents(None) == 0.0


def test_denominateur_toujours_complet() -> None:
    """Un champ manquant du dict compte comme non reconnu, pas comme hors calcul.

    Sinon le taux remonterait mécaniquement à mesure que l'extraction se dégrade.
    """
    partiel = {"total_ht": Decimal("1.0000"), "total_ttc": Decimal("1.0000")}

    assert taux_champs_reconnus(partiel) == 0.2  # 2 sur 10, pas 2 sur 2


# --- Contenu du run tracé --------------------------------------------------


def test_run_de_succes_contient_metriques_et_tags(store_mlflow: str) -> None:
    """Une extraction réussie produit un run complet : 5 métriques + 10 confiances."""
    track_extraction_quality(
        _payload_succes(),
        mode_extraction=ModeExtraction.PDF_NATIF,
        duree_secondes=1.5,
    )

    run = _dernier_run(store_mlflow)

    assert run.data.metrics["score_confiance"] == pytest.approx(0.9)
    assert run.data.metrics["taux_champs_reconnus"] == pytest.approx(1.0)
    assert run.data.metrics["taux_champs_presents"] == pytest.approx(1.0)
    assert run.data.metrics["extraction_reussie"] == pytest.approx(1.0)
    assert run.data.metrics["duree_secondes"] == pytest.approx(1.5)
    # Une métrique par champ suivi : c'est ce qui révèle les champs
    # chroniquement mal extraits.
    for champ in monitoring.TRACKED_FIELDS:
        assert run.data.metrics[f"confiance_{champ}"] == pytest.approx(1.0)

    assert run.data.tags["id_document"] == "42"
    assert run.data.tags["statut"] == "succes"
    assert run.data.tags["type_document"] == "facture"
    assert run.data.tags["mode_extraction"] == "pdf_natif"
    assert run.data.tags["modele_llm"] == settings.GROQ_MODEL
    assert run.data.tags["alerte"] == "false"


def test_run_d_echec_est_trace(store_mlflow: str) -> None:
    """Un échec est tracé aussi — c'est le signal le plus utile à suivre.

    Les confiances par champ sont omises (aucune n'a été calculée) plutôt que
    forcées à 0, ce qui ferait croire à des champs mal lus.
    """
    track_extraction_quality(
        _payload_echec(),
        mode_extraction=ModeExtraction.OCR,
        duree_secondes=0.2,
    )

    run = _dernier_run(store_mlflow)

    assert run.data.metrics["score_confiance"] == pytest.approx(0.0)
    assert run.data.metrics["extraction_reussie"] == pytest.approx(0.0)
    assert run.data.metrics["taux_champs_reconnus"] == pytest.approx(0.0)
    assert not any(clé.startswith("confiance_") for clé in run.data.metrics)

    assert run.data.tags["statut"] == "echec"
    assert run.data.tags["mode_extraction"] == "ocr"
    # Aucune suggestion de type n'a été produite : distinct du « le modèle a
    # répondu mais n'a pas su trancher » (TypeDocument.INCONNU).
    assert run.data.tags["type_document"] == "non_calcule"


def test_mode_inconnu_est_trace_tel_quel(store_mlflow: str) -> None:
    """Détection en échec → mode ``inconnu``, on ne devine pas le chemin."""
    track_extraction_quality(
        _payload_echec(),
        mode_extraction=ModeExtraction.INCONNU,
        duree_secondes=0.1,
    )

    assert _dernier_run(store_mlflow).data.tags["mode_extraction"] == "inconnu"


def test_aucune_donnee_sensible_n_est_tracee(store_mlflow: str) -> None:
    """Le run ne contient que des agrégats : aucun SIRET, IBAN, montant ni date.

    Garantie par construction (liste blanche dans ``monitoring.py``), vérifiée
    ici de bout en bout sur tout ce que le store a réellement écrit.
    """
    track_extraction_quality(
        _payload_succes(),
        mode_extraction=ModeExtraction.OCR,
        duree_secondes=1.0,
    )

    run = _dernier_run(store_mlflow)
    trace = " ".join(
        [
            *run.data.tags.keys(),
            *run.data.tags.values(),
            *run.data.metrics.keys(),
            *(str(valeur) for valeur in run.data.metrics.values()),
            *run.data.params.keys(),
            *run.data.params.values(),
        ]
    )

    for valeur in _VALEURS_SENSIBLES:
        assert valeur not in trace, f"donnée sensible tracée : {valeur}"
    # Les secrets ne doivent évidemment pas y être non plus.
    assert settings.SECRET_OCR_TOKEN not in trace
    assert settings.GROQ_API_KEY not in trace


# --- Alerte ----------------------------------------------------------------


def test_score_degrade_declenche_l_alerte(
    store_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Sous le seuil : WARNING applicatif + tag ``alerte`` filtrable dans l'UI.

    C'est le vecteur d'alerte du service, à défaut d'alerting natif dans MLflow.
    """
    payload = _payload_succes(score_confiance=Decimal("0.5000"))

    with caplog.at_level("WARNING"):
        track_extraction_quality(
            payload, mode_extraction=ModeExtraction.OCR, duree_secondes=1.0
        )

    assert "dégradée" in caplog.text
    assert "42" in caplog.text
    assert _dernier_run(store_mlflow).data.tags["alerte"] == "true"


def test_score_au_dessus_du_seuil_ne_declenche_rien(
    store_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Au-dessus du seuil : aucun bruit, tag ``alerte`` à ``false``."""
    with caplog.at_level("WARNING"):
        track_extraction_quality(
            _payload_succes(), mode_extraction=ModeExtraction.OCR, duree_secondes=1.0
        )

    assert "dégradée" not in caplog.text
    assert _dernier_run(store_mlflow).data.tags["alerte"] == "false"


# --- Garanties de non-nuisance ---------------------------------------------


def test_desactive_par_defaut_n_ecrit_rien(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Traçage désactivé (le défaut, notamment en CI) : rien n'est écrit."""
    store = tmp_path / "mlruns"
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", False)
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", str(store))

    track_extraction_quality(
        _payload_succes(), mode_extraction=ModeExtraction.OCR, duree_secondes=1.0
    )

    assert not store.exists()


def test_settings_de_test_desactivent_le_monitoring() -> None:
    """La suite tourne monitoring éteint : aucun test existant n'est perturbé."""
    assert settings.MLFLOW_ENABLED is False


def test_echec_du_tracage_ne_remonte_jamais(
    store_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Un traçage en panne est journalisé et avalé : le monitoring ne casse rien.

    L'extraction est terminée et déjà transmise à l'API data quand on arrive
    ici — un store injoignable n'est qu'une perte d'observabilité.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("store MLflow injoignable")

    monkeypatch.setattr(mlflow, "start_run", _boom)

    with caplog.at_level("WARNING"):
        track_extraction_quality(  # ne doit pas lever
            _payload_succes(), mode_extraction=ModeExtraction.OCR, duree_secondes=1.0
        )

    assert "traçage" in caplog.text
    assert "42" in caplog.text
