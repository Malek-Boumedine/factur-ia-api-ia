"""Point d'entrée de l'API IA d'extraction de factures."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.extractions.router import router as extractions_router

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
