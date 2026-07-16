"""Client du callback OCR : envoi du résultat d'extraction à l'API data.

Dernière étape du pipeline : POSTe le ``OcrWebhookPayload`` (succès comme échec)
vers le webhook OCR de l'API data (``settings.ocr_callback_url``), authentifié
par le header ``X-OCR-Secret-Token`` (secret partagé — jamais journalisé, ni en
clair ni dans les messages d'erreur).

Sérialisation via ``model_dump_json()`` (Pydantic v2) : les ``Decimal`` partent
en chaînes JSON — précision des montants préservée, reparsés en ``Decimal`` par
le schéma miroir côté API data — et les ``date`` en ISO 8601.

Stratégie de retry volontairement simple (pas de lib dédiée) :
``HTTP_MAX_RETRIES`` tentatives au total, en ne retentant que le transitoire
(timeout, erreur réseau, 5xx). Les réponses définitives (403 token refusé,
404 ``id_document`` inconnu, tout autre statut inattendu) échouent
immédiatement : rejouer donnerait le même résultat. Pas de backoff entre les
tentatives — dette MVP assumée, comme la perte du payload en cas d'échec
définitif (limite du choix ``BackgroundTasks``, cf. CLAUDE.md).
"""

import logging

import httpx

from src.callback.schemas import OcrWebhookPayload
from src.core.config import settings

logger = logging.getLogger(__name__)

# Réponses définitives connues du contrat, pour des logs explicites.
_DEFINITIVE_STATUS_DETAILS = {
    403: "token OCR partagé refusé par l'API data",
    404: "id_document inconnu côté API data",
}


class CallbackError(Exception):
    """Envoi du payload au webhook OCR de l'API data définitivement échoué.

    Levée dès une réponse définitive (403 token refusé, 404 ``id_document``
    inconnu, statut inattendu) ou après épuisement des tentatives sur du
    transitoire (timeout, réseau, 5xx). L'orchestrateur du pipeline l'attrape
    et se contente de journaliser : la tâche de fond ne doit jamais planter,
    et il n'existe pas (encore) de file de rejeu.
    """


def _build_client() -> httpx.Client:
    """Construit un client httpx éphémère (timeout de la configuration).

    Créé à la volée et non mis en cache au niveau module : un seul envoi par
    pipeline, ``httpx.Client`` est peu coûteux à instancier (contrairement au
    ``Reader`` EasyOCR ou au client Groq), et le ``with`` de l'appelant
    garantit la fermeture propre des connexions.
    """
    return httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS)


def _log_success(id_document: int, response: httpx.Response) -> None:
    """Journalise la confirmation renvoyée par l'API data (ids créés côté data).

    Purement informatif : un corps absent, non-JSON ou d'une forme inattendue
    (non-objet) donne un log dégradé, jamais une exception — le callback a été
    accepté, la tâche de fond ne doit pas planter sur de la journalisation.
    """
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        logger.info(
            "Document %s — callback OCR accepté (réponse sans corps JSON exploitable).",
            id_document,
        )
        return
    logger.info(
        "Document %s — callback OCR accepté "
        "(id_extraction=%s, id_facture=%s, statut=%s).",
        id_document,
        data.get("id_extraction"),
        data.get("id_facture"),
        data.get("statut"),
    )


def send_callback(payload: OcrWebhookPayload) -> None:
    """POSTe le résultat d'extraction au webhook OCR de l'API data.

    Le payload d'échec (``score_confiance = 0``) est envoyé exactement comme un
    payload de succès : c'est ainsi que l'API data apprend l'échec de
    l'extraction et passe le document en « erreur ».

    Args:
        payload: résultat d'extraction conforme au contrat (succès ou échec).

    Raises:
        CallbackError: envoi définitivement échoué — réponse définitive (403,
            404, statut inattendu) ou transitoire (timeout, réseau, 5xx)
            persistant après ``HTTP_MAX_RETRIES`` tentatives.
    """
    headers = {
        "X-OCR-Secret-Token": settings.SECRET_OCR_TOKEN,
        "Content-Type": "application/json",
    }
    body = payload.model_dump_json()
    last_failure = "aucune tentative effectuée"

    with _build_client() as client:
        for attempt in range(1, settings.HTTP_MAX_RETRIES + 1):
            try:
                response = client.post(
                    settings.ocr_callback_url,
                    content=body,
                    headers=headers,
                )
            except httpx.TransportError as exc:
                # Timeout ou erreur réseau/connexion : transitoire, on retente.
                last_failure = f"erreur réseau ou timeout ({type(exc).__name__})"
                logger.warning(
                    "Document %s — callback OCR, tentative %d/%d échouée : %s.",
                    payload.id_document,
                    attempt,
                    settings.HTTP_MAX_RETRIES,
                    last_failure,
                )
                continue

            if response.status_code == 200:
                _log_success(payload.id_document, response)
                return

            if response.status_code >= 500:
                # API data momentanément indisponible : transitoire, on retente.
                last_failure = f"HTTP {response.status_code}"
                logger.warning(
                    "Document %s — callback OCR, tentative %d/%d échouée : %s.",
                    payload.id_document,
                    attempt,
                    settings.HTTP_MAX_RETRIES,
                    last_failure,
                )
                continue

            # Réponse définitive : rejouer donnerait le même résultat.
            detail = _DEFINITIVE_STATUS_DETAILS.get(
                response.status_code, "réponse inattendue de l'API data"
            )
            logger.error(
                "Document %s — callback OCR refusé (HTTP %d : %s), abandon.",
                payload.id_document,
                response.status_code,
                detail,
            )
            raise CallbackError(
                f"Callback OCR refusé pour le document {payload.id_document} "
                f"(HTTP {response.status_code} : {detail})."
            )

    raise CallbackError(
        f"Envoi du callback OCR échoué pour le document {payload.id_document} "
        f"après {settings.HTTP_MAX_RETRIES} tentatives ({last_failure})."
    )
