"""Orchestrateur du pipeline d'extraction : du fichier reçu au payload contrat.

Câble les briques déjà écrites et testées en un seul enchaînement, exécuté en
tâche de fond (``fastapi.BackgroundTasks``) après le ``202`` de l'endpoint :

    routage extraction → texte brut → structuration LLM → score de confiance
    → validation contrat → ``OcrWebhookPayload``

Le module est **synchrone** à dessein : les briques lourdes (``pdfplumber``,
EasyOCR, client Groq) sont bloquantes. Lancé via ``BackgroundTasks``, il tourne
dans le threadpool de FastAPI, sans bloquer l'event loop (compromis MVP assumé,
cf. CLAUDE.md — une vraie file de tâches viendra plus tard).

Routage d'extraction (l'orchestrateur connaît le ``content_type``) :

- image (JPEG/PNG) → OCR direct (``extract_ocr_text(..., is_pdf=False)``), sans
  passer par le détecteur PDF ;
- PDF → ``detect_pdf_type`` puis, selon le résultat, extraction native
  (``pdfplumber``) ou OCR (``extract_ocr_text(..., is_pdf=True)``).

Gestion d'erreurs volontairement **minimale** ici : toutes les briques exposent
le même contrat d'échec (« extraction inexploitable → ``score_confiance = 0`` »),
donc un unique ``try/except`` englobant les traduit en payload d'échec via
``build_failure_payload``. La gestion fine de bout en bout (timeouts, causes
distinctes, échec du POST callback, retries) est une tâche ultérieure.
"""

import logging

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
    """Exécute le pipeline complet et construit le payload contrat à transmettre.

    Enchaîne extraction du texte, structuration LLM, calcul du score de confiance
    et validation contre ``OcrWebhookPayload``. En cas d'échec d'une brique
    (document illisible, appel LLM en erreur, JSON inexploitable, validation
    contrat impossible), renvoie le payload d'échec canonique
    (``score_confiance = 0``) plutôt que de propager l'exception : la tâche de
    fond ne doit jamais planter silencieusement, l'API data doit recevoir un
    verdict d'échec exploitable.

    Args:
        content: contenu binaire du fichier reçu (lu avant le ``202`` dans le
            router, car l'``UploadFile`` peut être fermé quand la tâche s'exécute).
        id_document: identifiant du document (vient de la requête).
        content_type: type MIME validé en amont (PDF, JPEG ou PNG).

    Returns:
        Un ``OcrWebhookPayload`` prêt à envoyer : soit l'extraction validée
        (``score_confiance > 0``), soit le payload d'échec (``score_confiance = 0``).
    """
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
    except _EXTRACTION_FAILURES:
        # Contrat d'échec commun aux briques : extraction inexploitable → payload
        # à ``score_confiance = 0``. Gestion fine (causes, timeouts) : tâche d'après.
        logger.warning(
            "Document %s — extraction inexploitable, payload d'échec émis.",
            id_document,
            exc_info=True,
        )
        return build_failure_payload(id_document)

    logger.info(
        "Document %s — extraction réussie (score de confiance : %s).",
        id_document,
        payload.score_confiance,
    )

    # TODO(callback): POST du payload vers le webhook OCR de l'API data
    # (callback.client -> settings.ocr_callback_url, header X-OCR-Secret-Token).
    # Tâche suivante. Pour l'instant le payload est construit puis renvoyé.
    return payload
