"""Orchestrateur du pipeline d'extraction : du fichier reçu au payload contrat.

Câble les briques déjà écrites et testées en un seul enchaînement, exécuté en
tâche de fond (``fastapi.BackgroundTasks``) après le ``202`` de l'endpoint :

    routage extraction → texte brut → structuration LLM → score de confiance
    → validation contrat → ``OcrWebhookPayload`` → envoi au callback OCR

Le module est **synchrone** à dessein : les briques lourdes (``pdfplumber``,
EasyOCR, client Groq) sont bloquantes. Lancé via ``BackgroundTasks``, il tourne
dans le threadpool de FastAPI, sans bloquer l'event loop (compromis MVP assumé,
cf. CLAUDE.md — une vraie file de tâches viendra plus tard).

Routage d'extraction (l'orchestrateur connaît le ``content_type``) :

- image (JPEG/PNG) → OCR direct (``extract_ocr_text(..., is_pdf=False)``), sans
  passer par le détecteur PDF ;
- PDF → ``detect_pdf_type`` puis, selon le résultat, extraction native
  (``pdfplumber``) ou OCR (``extract_ocr_text(..., is_pdf=True)``).

Gestion d'erreurs à deux niveaux : toutes les briques exposent le même contrat
d'échec (« extraction inexploitable → ``score_confiance = 0`` »), donc un
``try/except`` englobant traduit les exceptions métier connues en payload
d'échec via ``build_failure_payload`` (``WARNING``). Un filet ``except
Exception`` de dernier recours couvre l'inattendu — bug, ``MemoryError``,
exception d'une lib tierce non wrappée — et produit le même payload d'échec
(``ERROR``) : sans verdict envoyé, le document resterait bloqué « en attente »
pour toujours côté API data. Le payload — succès **ou** échec — est ensuite
POSTé au callback OCR de l'API data (retries gérés par ``callback.client``) ;
un envoi échoué, quelle qu'en soit la cause, est seulement journalisé, le
payload est perdu (limite assumée du choix ``BackgroundTasks``, cf. CLAUDE.md).
"""

import logging

from src.callback.client import CallbackError, send_callback
from src.callback.schemas import OcrWebhookPayload
from src.extractions.confidence import compute_confidence
from src.extractions.llm_client import LlmClientError
from src.extractions.ocr_extractor import OcrExtractionError, extract_ocr_text
from src.extractions.pdf_detector import (
    PdfDetectionError,
    PdfType,
    detect_pdf_type,
)
from src.extractions.pdf_extractor import PdfExtractionError, extract_native_pdf_text
from src.extractions.structurer import LlmStructurationError, structure_invoice
from src.extractions.validation import (
    PayloadValidationError,
    build_failure_payload,
    validate_extraction,
)

logger = logging.getLogger(__name__)

# Type MIME des PDF. Les autres types acceptés (image/jpeg, image/png) sont routés
# directement vers l'OCR par ``_extract_text``.
_PDF_MIME_TYPE = "application/pdf"

# Exceptions métier « extraction inexploitable » émises par les briques : toutes
# aboutissent au même payload d'échec (``score_confiance = 0``). Regroupées ici
# pour l'unique ``try/except`` englobant de l'orchestrateur.
_EXTRACTION_FAILURES = (
    PdfDetectionError,
    PdfExtractionError,
    OcrExtractionError,
    LlmClientError,
    LlmStructurationError,
    PayloadValidationError,
)


def _extract_text(content: bytes, content_type: str) -> str:
    """Extrait le texte brut selon le type de document (routage des trois chemins).

    Une image va directement à l'OCR (pas de détection). Un PDF passe par
    ``detect_pdf_type`` puis, selon sa nature, l'extraction native ou l'OCR.

    Args:
        content: contenu binaire du fichier reçu.
        content_type: type MIME validé en amont (PDF, JPEG ou PNG).

    Returns:
        Le texte brut extrait, prêt pour la structuration LLM.

    Raises:
        PdfDetectionError | PdfExtractionError | OcrExtractionError: document
            illisible ou sans texte exploitable (traduit en échec par l'appelant).
    """
    if content_type != _PDF_MIME_TYPE:
        # Image (JPEG/PNG) : OCR direct, sans passer par le détecteur PDF.
        return extract_ocr_text(content, is_pdf=False)

    pdf_type = detect_pdf_type(content)
    if pdf_type is PdfType.NATIVE:
        return extract_native_pdf_text(content)
    return extract_ocr_text(content, is_pdf=True)


def run_extraction_pipeline(
    content: bytes,
    id_document: int,
    content_type: str,
) -> OcrWebhookPayload:
    """Exécute le pipeline complet et transmet le payload contrat au callback.

    Enchaîne extraction du texte, structuration LLM, calcul du score de confiance
    et validation contre ``OcrWebhookPayload``. En cas d'échec d'une brique
    (document illisible, appel LLM en erreur, JSON inexploitable, validation
    contrat impossible) — ou de toute exception inattendue (filet de dernier
    recours) — produit le payload d'échec canonique (``score_confiance = 0``)
    plutôt que de propager l'exception : la tâche de fond ne doit jamais planter
    silencieusement, l'API data doit recevoir un verdict d'échec exploitable.

    Le payload — succès **ou** échec — est ensuite POSTé au callback OCR de
    l'API data (c'est le payload d'échec qui fait passer le document en
    « erreur » côté data). Un envoi définitivement échoué (``CallbackError``)
    est journalisé puis avalé : le payload est perdu, limite assumée du choix
    ``BackgroundTasks`` (cf. CLAUDE.md).

    Args:
        content: contenu binaire du fichier reçu (lu avant le ``202`` dans le
            router, car l'``UploadFile`` peut être fermé quand la tâche s'exécute).
        id_document: identifiant du document (vient de la requête).
        content_type: type MIME validé en amont (PDF, JPEG ou PNG).

    Returns:
        L'``OcrWebhookPayload`` envoyé au callback : soit l'extraction validée
        (``score_confiance > 0``), soit le payload d'échec (``score_confiance = 0``).
    """
    # Log de début : repérer les « début sans fin » (tâche qui pend ou meurt sans
    # verdict), le pipeline pouvant être long (OCR + LLM).
    logger.info(
        "Document %s — extraction démarrée (%s).",
        id_document,
        content_type,
    )

    try:
        raw_text = _extract_text(content, content_type)
        structured = structure_invoice(raw_text)

        # ``type_document`` est une suggestion IA interne, HORS contrat : jamais
        # transmise au callback, seulement journalisée (décision finale à l'humain).
        logger.info(
            "Document %s — type suggéré (interne, non transmis) : %s",
            id_document,
            structured["type_document"],
        )

        facture = structured["facture"]
        confidence = compute_confidence(facture)
        payload = validate_extraction(id_document, facture, confidence.score_global)
        logger.info(
            "Document %s — extraction réussie (score de confiance : %s).",
            id_document,
            payload.score_confiance,
        )
    except _EXTRACTION_FAILURES:
        # Contrat d'échec commun aux briques : extraction inexploitable → payload
        # à ``score_confiance = 0``. Gestion fine (causes, timeouts) : tâche d'après.
        logger.warning(
            "Document %s — extraction inexploitable, payload d'échec émis.",
            id_document,
            exc_info=True,
        )
        payload = build_failure_payload(id_document)
    except Exception:
        # Filet de dernier recours (bug, MemoryError, exception d'une lib tierce
        # non wrappée) : sans verdict envoyé, le document resterait bloqué « en
        # attente » pour toujours côté API data. ERROR (vs WARNING métier) : un
        # échec inattendu est probablement un bug et doit faire du bruit.
        logger.error(
            "Document %s — erreur inattendue dans le pipeline, payload d'échec émis.",
            id_document,
            exc_info=True,
        )
        payload = build_failure_payload(id_document)

    # Envoi au callback dans TOUS les cas : le payload d'échec doit remonter à
    # l'API data pour passer le document en « erreur ». Un échec définitif de
    # l'envoi est seulement journalisé : personne n'attend la tâche de fond et
    # il n'y a pas de file de rejeu — le payload est perdu (limite assumée du
    # choix BackgroundTasks, cf. CLAUDE.md).
    try:
        send_callback(payload)
    except CallbackError:
        logger.error(
            "Document %s — envoi au callback définitivement échoué, payload perdu.",
            id_document,
            exc_info=True,
        )
    except Exception:
        # Même filet que le pipeline : une erreur inattendue pendant l'envoi ne
        # doit pas faire planter la tâche de fond.
        logger.error(
            "Document %s — erreur inattendue pendant l'envoi au callback, "
            "payload perdu.",
            id_document,
            exc_info=True,
        )

    return payload
