"""Prompt système et json_schema de la structuration de facture.

Regroupe « ce qu'on demande au modèle » pour la structuration : le prompt système
(consignes FR) et le ``response_format`` structured outputs strict passé au client
Groq. La logique d'appel/parse vit dans ``structurer.py`` ; ce module ne contient
que des constantes.

Le schéma est le **miroir exact** du sous-ensemble « données extraites » de
``OcrWebhookPayload`` (``src/callback/schemas.py``) : mêmes noms de champs, même
arborescence. ``id_document`` (vient de la requête) et ``score_confiance`` (calculé
plus tard) en sont volontairement absents — ils ne sont pas produits par le LLM.

Divergence volontaire sur les totaux : ``total_ht`` / ``total_tva`` / ``total_ttc``
sont **non-nullables** dans ``OcrWebhookPayload``, mais **nullables ici** (``null``
autorisé). Raison : en structured output strict, un type ``number`` seul forcerait
le modèle à *inventer* un montant absent/illisible — à proscrire. Un total illisible
doit donc ressortir ``null``, jamais fabriqué. La réconciliation de cette divergence
(``null`` → échec ``score_confiance = 0`` ou correction human-in-the-loop) est du
ressort de la tâche validation/score suivante, pas de ce module.

Contraintes du mode ``strict: true`` (Groq) respectées par le schéma : tous les
champs listés dans ``required``, ``additionalProperties: false`` sur chaque objet,
et champs optionnels exprimés par une union avec ``null`` (``["string", "null"]``).
"""

from typing import Any

# Prompt système : consignes d'extraction. En français (destiné au modèle), il
# insiste sur les points sensibles constatés (taux TVA confondu avec un id,
# montants mal formatés, champs inventés, confusion émetteur/destinataire).
SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans l'extraction de données de factures françaises.
On te fournit le texte brut d'une seule facture (issu d'une extraction PDF ou d'un
OCR, parfois imparfait). Tu dois en extraire les données et répondre UNIQUEMENT par
un objet JSON conforme au schéma imposé, sans aucun texte ni commentaire autour.

Règles impératives :

- N'INVENTE JAMAIS de valeur. Si une information est absente, illisible ou
  incertaine, mets `null` (pour les champs qui l'autorisent). Il vaut toujours mieux
  `null` qu'une valeur devinée. Cela vaut aussi pour les totaux.
- `taux_tva` est un TAUX DE TVA EN POURCENTAGE (par exemple `20.00`, `10.00`,
  `5.50`, `0.00`), jamais un identifiant, un code, ni un montant de TVA en euros.
- Les montants (`total_ht`, `total_tva`, `total_ttc`, `prix_unitaire_ht`) sont des
  nombres décimaux : point comme séparateur décimal, sans séparateur de milliers,
  sans symbole monétaire ni texte (écris `1234.56`, pas `1 234,56 €`).
- `date_emission` est au format ISO `AAAA-MM-JJ` (par exemple `2026-07-06`).
- Distingue bien l'ÉMETTEUR du DESTINATAIRE : `siret_emetteur` est le SIRET du
  vendeur / prestataire qui émet la facture ; `siret_destinataire` est le SIRET du
  client facturé. Ne les intervertis pas.
- `lignes` : une entrée par ligne d'article ou de prestation. Pour chaque ligne,
  `quantite` vaut `1` si elle n'est pas précisée. S'il n'y a aucune ligne
  identifiable, renvoie une liste vide.
- `iban` : l'IBAN de paiement s'il figure sur la facture, sinon `null`.

Réponds seulement avec le JSON."""

# Nom du schéma transmis à Groq (identifiant libre, pas un champ de la facture).
_SCHEMA_NAME = "facture_extraite"

# Schéma d'une ligne de facture : miroir de ``LigneOcr``. Tous les champs sont
# non-nullables (une ligne sans désignation/prix/taux n'est pas une ligne). En
# strict mode, tous doivent figurer dans ``required``.
_LIGNE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "designation": {"type": "string"},
        "quantite": {"type": "number"},
        "prix_unitaire_ht": {"type": "number"},
        "taux_tva": {"type": "number"},  # pourcentage, ex. 20.00
    },
    "required": ["designation", "quantite", "prix_unitaire_ht", "taux_tva"],
}

# Schéma de la facture : miroir du sous-ensemble « données extraites » de
# ``OcrWebhookPayload``. Nullabilité alignée sur le contrat, SAUF les totaux
# (nullables ici, cf. divergence volontaire documentée en tête de module).
_INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "siret_emetteur": {"type": ["string", "null"]},
        "siret_destinataire": {"type": ["string", "null"]},
        "numero_facture": {"type": ["string", "null"]},
        "date_emission": {"type": ["string", "null"]},  # ISO AAAA-MM-JJ
        "total_ht": {"type": ["number", "null"]},
        "total_tva": {"type": ["number", "null"]},
        "total_ttc": {"type": ["number", "null"]},
        "iban": {"type": ["string", "null"]},
        "lignes": {"type": "array", "items": _LIGNE_SCHEMA},
    },
    "required": [
        "siret_emetteur",
        "siret_destinataire",
        "numero_facture",
        "date_emission",
        "total_ht",
        "total_tva",
        "total_ttc",
        "iban",
        "lignes",
    ],
}

# ``response_format`` complet à passer à ``call_llm`` (structured outputs strict).
INVOICE_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": _SCHEMA_NAME,
        "strict": True,
        "schema": _INVOICE_SCHEMA,
    },
}
