"""Monitoring de la qualité d'extraction : un run MLflow par extraction.

Module transverse (pendant du ``core/telemetry.py`` de l'API data, qui couvre lui
le monitoring *applicatif*). Ici on ne surveille pas « est-ce que l'app tourne ? »
mais « est-ce que le modèle d'extraction répond bien ? » : on trace dans le temps
les signaux de qualité que le pipeline produit déjà (score de confiance global,
confiance par champ, type de document suggéré, succès/échec), pour suivre la
qualité, détecter une dérive et la restituer.

Pourquoi MLflow plutôt que Prometheus : c'est l'outil de suivi *de modèle*, et il
sait ce qu'un système de métriques ne sait pas faire — attacher à chaque
extraction son identifiant et le **nom du modèle LLM utilisé**, donc répondre à
« le score a-t-il chuté quand j'ai changé de modèle ? ». En écriture il n'exige
aucun serveur : le store par défaut est un simple fichier SQLite local
(``MLFLOW_TRACKING_URI="sqlite:///mlflow.db"``) ; l'interface ne sert qu'à
*relire*.

Deux garanties, dans cet ordre de priorité :

1. **Le monitoring ne casse jamais l'extraction.** Traçage désactivé → retour
   immédiat, ``mlflow`` n'est même pas importé (import paresseux dans
   ``_log_run``) : coût strictement nul et comportement inchangé, en local comme
   en CI. Traçage actif → tout le corps est sous un ``try/except Exception`` qui
   journalise en ``WARNING`` et n'ose rien remonter. Disque plein, store
   illisible, bug du client MLflow : l'appelant ne le voit pas passer.
2. **Aucune donnée sensible ne sort d'ici.** Le contenu tracé est construit
   depuis une **liste blanche explicite** (``TRACKED_FIELDS`` pour les
   confiances, dictionnaires littéraux pour le reste) : on ne sérialise jamais
   le payload, on n'itère jamais sur ses champs. Ne partent que des agrégats —
   des nombres entre 0 et 1, une durée, des étiquettes catégorielles à valeurs
   bornées, l'``id_document``. Ne partent **jamais** : texte brut du document,
   SIRET, IBAN, numéro de facture, montants, dates, désignations de lignes, nom
   de fichier, ni aucun secret.

Compromis assumé — **l'alerte**. MLflow ne sait pas alerter. À défaut, un score
sous ``MONITORING_SEUIL_ALERTE`` déclenche ici un ``WARNING`` applicatif *et*
pose le tag ``alerte=true`` sur le run, qui rend les extractions dégradées
filtrables en un clic dans l'interface. C'est le vecteur d'alerte du service ;
une vraie règle d'alerting (Grafana) viendra avec la centralisation des logs.
"""

import logging
from decimal import Decimal
from enum import StrEnum
from typing import Final

from src.callback.schemas import OcrWebhookPayload
from src.core.config import settings

logger = logging.getLogger(__name__)


class ModeExtraction(StrEnum):
    """Chemin d'extraction du texte emprunté par le pipeline (valeurs du tag).

    Variable explicative la plus forte du score de confiance : une baisse de la
    qualité moyenne s'explique bien plus souvent par « davantage de documents
    scannés arrivent » que par une dégradation du modèle. Sans ce tag, on voit
    la dérive sans pouvoir l'expliquer.

    ``INCONNU`` couvre le cas où le chemin n'a pas pu être déterminé — détection
    du type de PDF en échec : on ne sait alors honnêtement pas par où l'on
    serait passé.
    """

    PDF_NATIF = "pdf_natif"
    OCR = "ocr"
    INCONNU = "inconnu"


# Liste blanche des champs suivis : les 10 champs du contrat sur lesquels
# ``confidence.py`` produit une confiance. Sert à la fois de dénominateur aux
# taux et de source des métriques ``confiance_<champ>`` — on lit ces clés dans
# ``par_champ``, on n'itère jamais sur le payload.
TRACKED_FIELDS: Final[tuple[str, ...]] = (
    "siret_emetteur",
    "siret_destinataire",
    "numero_facture",
    "date_emission",
    "date_echeance",
    "total_ht",
    "total_tva",
    "total_ttc",
    "iban",
    "lignes",
)

# Seuil à partir duquel un champ est considéré comme « reconnu » (extrait avec
# une confiance suffisante). Valeur adossée à ``_UNVERIFIED_PRESENT`` de
# ``confidence.py`` : à 0.7, un champ est présent et non démenti par un contrôle
# d'intégrité. En dessous, il est soit absent (0), soit mal formé (0.2), soit
# invalidé par un contrôle (0.4). Le seuil n'est donc pas arbitraire : il colle
# à la sémantique du scoring existant.
_SEUIL_CHAMP_RECONNU: Final = Decimal("0.7")

# Seuil de simple *présence* d'un champ. Les scores sont quantifiés à 4
# décimales par ``confidence.py`` (``_QUANTUM``) : « strictement positif »
# équivaut donc à « ≥ 0.0001 », ce qui permet de réutiliser la même mécanique de
# comparaison que pour le seuil de fiabilité.
_SEUIL_CHAMP_PRESENT: Final = Decimal("0.0001")

# Valeur du tag ``type_document`` quand aucune suggestion n'a été produite
# (payload d'échec). Distincte de ``TypeDocument.INCONNU``, qui signifie « le
# modèle a répondu, mais n'a pas su trancher » — deux situations différentes
# qu'on ne veut pas confondre à la lecture.
_TYPE_NON_CALCULE: Final = "non_calcule"


def _part_des_champs(par_champ: dict[str, Decimal] | None, seuil: Decimal) -> float:
    """Part des champs suivis dont la confiance atteint ``seuil`` (0.0 à 1.0).

    Le dénominateur est toujours ``len(TRACKED_FIELDS)`` : un champ absent de
    ``par_champ`` compte comme non atteint, jamais comme « hors calcul » — sinon
    le taux remonterait mécaniquement quand l'extraction se dégrade.

    ``par_champ`` vaut ``None`` sur un payload d'échec : le taux est alors 0.0,
    ce qui est exact (aucun champ reconnu) et garde la série continue.
    """
    if not par_champ:
        return 0.0
    atteints = sum(
        1 for champ in TRACKED_FIELDS if par_champ.get(champ, Decimal("0")) >= seuil
    )
    return atteints / len(TRACKED_FIELDS)


def taux_champs_reconnus(par_champ: dict[str, Decimal] | None) -> float:
    """Taux de champs extraits avec une confiance suffisante (≥ 0.7)."""
    return _part_des_champs(par_champ, _SEUIL_CHAMP_RECONNU)


def taux_champs_presents(par_champ: dict[str, Decimal] | None) -> float:
    """Taux de champs simplement extraits, quelle que soit leur fiabilité (> 0).

    Complément du taux précédent : l'écart entre les deux distingue « le champ
    manque » de « le champ est là mais douteux » — deux problèmes qui appellent
    des corrections différentes (prompt vs qualité de la source).
    """
    return _part_des_champs(par_champ, _SEUIL_CHAMP_PRESENT)


def _build_metrics(
    payload: OcrWebhookPayload, duree_secondes: float
) -> dict[str, float]:
    """Construit les métriques numériques du run (liste blanche, agrégats seuls).

    Les ``Decimal`` sont convertis en ``float`` — MLflow ne stocke que des
    flottants. La précision exacte des scores n'a pas d'enjeu ici : elle est
    préservée là où elle compte, dans le payload envoyé à l'API data.
    """
    metrics: dict[str, float] = {
        "score_confiance": float(payload.score_confiance),
        "taux_champs_reconnus": taux_champs_reconnus(payload.par_champ),
        "taux_champs_presents": taux_champs_presents(payload.par_champ),
        # ``score_confiance == 0`` est le marqueur d'échec du contrat (cf. CLAUDE.md).
        "extraction_reussie": 0.0 if payload.score_confiance == 0 else 1.0,
        "duree_secondes": duree_secondes,
    }

    # Confiance champ par champ : c'est ce qui révèle les champs chroniquement
    # mal extraits, et donc ce qui pilote les retouches du prompt. Omises sur un
    # payload d'échec (aucune confiance calculée) plutôt que forcées à 0, qui
    # ferait croire à un champ mal lu alors qu'il n'y a pas eu de lecture.
    if payload.par_champ is not None:
        for champ in TRACKED_FIELDS:
            score = payload.par_champ.get(champ)
            if score is not None:
                metrics[f"confiance_{champ}"] = float(score)

    return metrics


def _build_tags(
    payload: OcrWebhookPayload,
    mode_extraction: ModeExtraction,
    alerte: bool,
) -> dict[str, str]:
    """Construit les étiquettes du run (dimensions de filtrage et de regroupement).

    Toutes à valeurs bornées, sauf ``id_document`` — un entier interne, sans
    valeur en soi, mais indispensable pour retrouver le document derrière un run
    dégradé. ``modele_llm`` est la dimension qui permet de comparer deux modèles
    Groq sur la même population de documents.
    """
    return {
        "id_document": str(payload.id_document),
        "statut": "echec" if payload.score_confiance == 0 else "succes",
        "type_document": payload.type_document or _TYPE_NON_CALCULE,
        "mode_extraction": mode_extraction.value,
        "modele_llm": settings.GROQ_MODEL,
        "alerte": "true" if alerte else "false",
    }


def _log_run(
    payload: OcrWebhookPayload,
    mode_extraction: ModeExtraction,
    duree_secondes: float,
) -> None:
    """Enregistre un run MLflow pour cette extraction (appelé sous protection).

    L'import de ``mlflow`` est fait ici, et non au niveau module : quand le
    traçage est désactivé (le défaut, notamment en CI), la bibliothèque n'est
    jamais chargée.
    """
    import mlflow

    alerte = payload.score_confiance < settings.MONITORING_SEUIL_ALERTE
    if alerte:
        # Vecteur d'alerte du service, à défaut d'alerting natif dans MLflow.
        logger.warning(
            "Document %s — qualité d'extraction dégradée "
            "(score %s, seuil d'alerte %s).",
            payload.id_document,
            payload.score_confiance,
            settings.MONITORING_SEUIL_ALERTE,
        )

    # URI et expérience posés explicitement : la bibliothèque ne lit pas le
    # fichier .env (chargé par pydantic-settings, sans export dans os.environ).
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"document-{payload.id_document}"):
        mlflow.set_tags(_build_tags(payload, mode_extraction, alerte))
        mlflow.log_metrics(_build_metrics(payload, duree_secondes))


def track_extraction_quality(
    payload: OcrWebhookPayload,
    *,
    mode_extraction: ModeExtraction,
    duree_secondes: float,
) -> None:
    """Trace la qualité d'une extraction — succès comme échec — dans MLflow.

    Point d'entrée unique du monitoring qualité : le pipeline l'appelle une
    fois, après l'envoi au callback. Ne lève jamais, ne bloque jamais l'appelant
    (cf. les deux garanties en tête de module) et ne fait rien du tout tant que
    ``MLFLOW_ENABLED`` est faux.

    Args:
        payload: résultat d'extraction déjà envoyé à l'API data. Lu seulement —
            score global, confiance par champ, type suggéré — jamais sérialisé.
        mode_extraction: chemin d'extraction emprunté (PDF natif, OCR, inconnu).
        duree_secondes: durée du pipeline complet, callback compris.
    """
    if not settings.MLFLOW_ENABLED:
        return

    try:
        _log_run(payload, mode_extraction, duree_secondes)
    except Exception:
        # Le monitoring ne doit jamais casser la fonctionnalité : l'extraction
        # est terminée et déjà transmise à l'API data, un traçage en échec n'est
        # qu'une perte d'observabilité.
        logger.warning(
            "Document %s — traçage de la qualité d'extraction échoué, ignoré.",
            payload.id_document,
            exc_info=True,
        )
