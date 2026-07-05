from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class LigneOcr(BaseModel):
    designation: str
    quantite: Decimal = Decimal("1.0")
    prix_unitaire_ht: Decimal
    taux_tva: Decimal


class OcrWebhookPayload(BaseModel):
    id_document: int
    score_confiance: Decimal
    siret_emetteur: str | None = None
    siret_destinataire: str | None = None
    numero_facture: str | None = None
    date_emission: date | None = None
    total_ht: Decimal
    total_tva: Decimal
    total_ttc: Decimal
    iban: str | None = None
    lignes: list[LigneOcr] = []
