"""Sécurité : vérification du token partagé entre l'API data et l'API IA."""

from fastapi import Header, HTTPException, status

from src.core.config import settings


async def verify_ocr_token(
    x_ocr_secret_token: str = Header(..., alias="X-OCR-Secret-Token"),
) -> None:
    """Vérifie le token partagé sur les appels entrants depuis l'API data.

    Lève un 403 si le token est absent ou invalide.
    """
    if x_ocr_secret_token != settings.SECRET_OCR_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token OCR invalide ou manquant.",
        )