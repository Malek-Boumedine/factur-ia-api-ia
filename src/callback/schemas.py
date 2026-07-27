from datetime import date
from decimal import Decimal
from typing import Literal

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
    # Extension additive du contrat (optionnels, ``None`` = non calculé — payload
    # d'échec ou version antérieure du service) : le type de document suggéré par
    # l'IA (décision finale à l'humain) et la confiance par champ (clés = noms des
    # champs ci-dessus, scores 0-1 quantifiés à 4 décimales) pour le surlignage
    # des champs douteux côté front.
    type_document: Literal["devis", "facture", "avoir", "inconnu"] | None = None
    par_champ: dict[str, Decimal] | None = None
