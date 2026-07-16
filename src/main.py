"""Point d'entrée de l'API IA d'extraction de factures."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.extractions.router import router as extractions_router


def _configure_logging() -> None:
    """Configure le logging applicatif (uvicorn ne configure que SES loggers).

    Sans cette configuration, le logger racine n'a aucun handler : les logs
    ``INFO`` du pipeline (extraction démarrée/réussie, callback accepté) sont
    perdus, et les ``WARNING``/``ERROR`` sortent bruts (handler de dernier
    recours de Python), sans horodatage ni nom de logger — diagnostic impossible
    en production. ``basicConfig`` est sans effet si le logger racine a déjà des
    handlers (configuration posée par le déployeur) : on n'écrase rien.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


_configure_logging()

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(extractions_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Vérifie que le service est en ligne."""
    return {"status": "ok"}
