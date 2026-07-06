"""Schémas du slice extractions (réception de documents)."""

from pydantic import BaseModel


class ExtractionAccepted(BaseModel):
    """Accusé de réception d'un document à traiter (réponse 202)."""

    id_document: int
    message: str = "Document reçu, traitement en cours."
