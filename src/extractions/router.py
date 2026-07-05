"""Endpoint de réception des documents à extraire.

Contrat figé avec l'API data : `POST /extractions`, `multipart/form-data`
avec `file` + `id_document`, protégé par le header `X-OCR-Secret-Token`.

Cette version se limite à la réception : validation du type et de la taille
du fichier, puis accusé de réception `202`. Le pipeline d'extraction
(OCR / LLM / callback) sera branché dans une tâche ultérieure.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.core.config import settings
from src.core.security import verify_ocr_token
from src.extractions.schemas import ExtractionAccepted

# Types MIME acceptés, cohérents avec ce que transmet l'API data.
ALLOWED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
)

# Taille de lecture par bloc pour mesurer le fichier sans tout charger en mémoire.
_CHUNK_SIZE = 1024 * 1024  # 1 Mo

router = APIRouter(tags=["extractions"])


async def _validate_upload(file: UploadFile) -> None:
    """Valide le type MIME (400) puis la taille (413) du fichier reçu.

    La taille est mesurée par lecture en blocs avec abandon anticipé dès
    dépassement, sans charger tout le contenu en mémoire. Le fichier est
    rembobiné (`seek(0)`) pour rester exploitable par la suite du pipeline.
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("Type de fichier non supporté. Formats acceptés : PDF, JPEG, PNG."),
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total = 0
    while chunk := await file.read(_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            await file.seek(0)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "Fichier trop volumineux "
                    f"(maximum {settings.MAX_UPLOAD_SIZE_MB} Mo)."
                ),
            )

    await file.seek(0)


@router.post(
    "/extractions",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_ocr_token)],
    summary="Réception d'un document à extraire",
)
async def receive_extraction(
    file: UploadFile = File(...),
    id_document: int = Form(...),
) -> ExtractionAccepted:
    """Reçoit un document, le valide et accuse réception.

    Renvoie `202` immédiatement (traitement asynchrone prévu). Le token
    `X-OCR-Secret-Token` est vérifié en amont par la dépendance.
    """
    await _validate_upload(file)

    # TODO(pipeline): déclencher l'orchestrateur d'extraction en tâche de fond
    # (fastapi.BackgroundTasks -> extractions.service), qui poussera le
    # résultat vers le callback de l'API data. Le contenu de `file` (rembobiné)
    # sera transmis ici. Rien n'est traité pour l'instant.

    return ExtractionAccepted(id_document=id_document)
