"""Endpoint de réception des documents à extraire.

Contrat figé avec l'API data : `POST /extractions`, `multipart/form-data`
avec `file` + `id_document`, protégé par le header `X-OCR-Secret-Token`.

Réception : validation du type et de la taille du fichier, accusé de réception
`202` immédiat, puis extraction déclenchée en tâche de fond
(`fastapi.BackgroundTasks`, compromis MVP assumé — cf. CLAUDE.md).
"""

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from src.core.config import settings
from src.core.security import verify_ocr_token
from src.extractions.schemas import ExtractionAccepted
from src.extractions.service import run_extraction_pipeline

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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    id_document: int = Form(...),
) -> ExtractionAccepted:
    """Reçoit un document, le valide, accuse réception et déclenche l'extraction.

    Renvoie `202` immédiatement, puis l'extraction est traitée en tâche de fond.
    Le token `X-OCR-Secret-Token` est vérifié en amont par la dépendance.

    Le contenu est lu ici, avant le `202` : l'`UploadFile` (fichier temporaire)
    peut être fermé quand la tâche de fond s'exécute, on transmet donc des octets
    immuables à l'orchestrateur.
    """
    await _validate_upload(file)

    # content_type garanti non-null et dans ALLOWED_MIME_TYPES par _validate_upload.
    content = await file.read()
    content_type = file.content_type or ""

    background_tasks.add_task(
        run_extraction_pipeline,
        content,
        id_document,
        content_type,
    )

    return ExtractionAccepted(id_document=id_document)
