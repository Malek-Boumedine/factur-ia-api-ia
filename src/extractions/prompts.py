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

Le schéma effectivement envoyé au LLM (``INVOICE_JSON_SCHEMA``) ajoute au miroir du
contrat deux champs hors contrat, produits dans le même appel LLM puis séparés du
sous-ensemble « données extraites » côté ``structurer.py`` :

- ``type_document`` (suggestion de classification devis/facture/avoir/inconnu), qui
  rejoint le contrat plus loin dans le pipeline via le champ optionnel
  ``type_document`` d'``OcrWebhookPayload`` ;
- ``delai_paiement_jours`` (échéance exprimée en délai — « net 30 », « paiement à
  30 jours »), qui ne rejoint jamais le contrat : ``structurer.py`` le consomme pour
  dériver ``date_echeance`` (émission + délai) quand aucune date d'échéance absolue
  n'a été lue. L'arithmétique est faite en Python, pas par le modèle : déterministe
  et testable.

Le miroir pur du sous-ensemble « données extraites » reste ``_INVOICE_SCHEMA``.
"""

from enum import StrEnum
from typing import Any


class TypeDocument(StrEnum):
    """Nature du document détectée par l'IA — suggestion non contraignante.

    La décision finale revient à l'humain (validation human-in-the-loop côté API
    data / front). Transmise au callback via le champ optionnel ``type_document``
    d'``OcrWebhookPayload`` (le ``Literal`` du contrat doit rester synchronisé avec
    ces valeurs). Source unique des valeurs autorisées du champ ``type_document``
    du schéma LLM.
    """

    DEVIS = "devis"
    FACTURE = "facture"
    AVOIR = "avoir"
    INCONNU = "inconnu"  # type indéterminable : valeur par défaut


# Prompt système : consignes d'extraction. En français (destiné au modèle), il
# insiste sur les points sensibles constatés (taux TVA confondu avec un id,
# montants mal formatés, séparateur de milliers français tronqué — « 1 850,00 »
# lu 850 —, champs inventés, confusion émetteur/destinataire).
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
- LECTURE des montants : le texte source utilise le format français des nombres.
  L'espace entre groupes de chiffres (espace normal, insécable ou fine) est un
  séparateur de milliers : agrège les groupes en un seul nombre. La virgule est le
  séparateur décimal. Exemples : « 1 850,00 » vaut `1850.00` ; « 12 345,67 » vaut
  `12345.67` ; « 850,00 » vaut `850.00`. Ne tronque JAMAIS les chiffres situés
  avant un espace : dans une ligne de tableau, un chiffre isolé devant un groupe
  de chiffres est souvent le début du montant (le millier), pas une quantité.
- ÉCRITURE des montants (`total_ht`, `total_tva`, `total_ttc`,
  `prix_unitaire_ht`) : nombres décimaux avec le point comme séparateur décimal,
  sans séparateur de milliers, sans symbole monétaire ni texte (écris `1850.00`,
  pas `1 850,00 €`).
- `date_emission` et `date_echeance` sont au format ISO `AAAA-MM-JJ` (par exemple
  `2026-07-06`). Le document, lui, utilise les formats français : convertis-les
  (« 06/07/2026 » et « 06-07-2026 » valent `2026-07-06` ; « 6 juillet 2026 » vaut
  `2026-07-06` ; « 15 mars 2026 » vaut `2026-03-15`). Dans une date française en
  chiffres, le JOUR précède le MOIS : « 03/04/2026 » est le 3 avril 2026, jamais le
  4 mars.
- NE CONFONDS PAS les deux dates :
  - `date_emission` est la date à laquelle la facture est établie (« date »,
    « date de facture », « émise le », « fait le », « le … »).
  - `date_echeance` est la DATE LIMITE DE PAIEMENT (« échéance », « date
    d'échéance », « à régler avant le … », « payable avant le … », « à payer au
    plus tard le … », « date limite de paiement »). Elle est postérieure ou égale
    à la date d'émission.
  Si le document ne porte qu'une seule date, c'est la date d'émission : mets
  `date_echeance` à `null`. Ne recopie JAMAIS la date d'émission dans
  `date_echeance`.
- `delai_paiement_jours` : quand l'échéance est exprimée en DÉLAI et non en date
  (« paiement à 30 jours », « net 30 », « règlement sous 45 jours », « payable à
  60 jours »), mets ici le NOMBRE DE JOURS (`30`, `45`, `60`) ; sinon `null`. Ne
  calcule pas la date toi-même, le délai est converti en date en aval. Si le
  document porte une date d'échéance explicite, renseigne `date_echeance` — et
  aussi `delai_paiement_jours` si le délai est écrit à côté.
- Distingue bien l'ÉMETTEUR du DESTINATAIRE : `siret_emetteur` est le SIRET du
  vendeur / prestataire qui émet la facture ; `siret_destinataire` est le SIRET du
  client facturé. Ne les intervertis pas.
- `lignes` : une entrée par ligne d'article ou de prestation. Pour chaque ligne,
  `quantite` vaut `1` si elle n'est pas précisée. S'il n'y a aucune ligne
  identifiable, renvoie une liste vide.
- `iban` : l'IBAN de paiement s'il figure sur la facture, sinon `null`.
- `type_document` : classe la nature du document parmi `facture`, `devis`, `avoir`
  ou `inconnu`. Indices : un `devis` propose un prix avant commande (« devis »,
  « proposition commerciale », pas de paiement dû) ; un `avoir` est une note de
  crédit / remboursement (« avoir », « note de crédit », montants négatifs) ; une
  `facture` réclame un paiement (« facture », « à payer »). Si aucun indice fiable,
  mets `inconnu` — ne devine pas. Ce champ est une simple suggestion.

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
        "date_echeance": {"type": ["string", "null"]},  # ISO AAAA-MM-JJ
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
        "date_echeance",
        "total_ht",
        "total_tva",
        "total_ttc",
        "iban",
        "lignes",
    ],
}

# Schéma effectivement envoyé au LLM : miroir du contrat (``_INVOICE_SCHEMA``)
# augmenté, à plat, des deux champs hors contrat — ``type_document`` (suggestion de
# classification) et ``delai_paiement_jours`` (échéance exprimée en délai, convertie
# en date côté Python). Un schéma plat est plus fiable pour le modèle qu'une
# imbrication ; la séparation hors-contrat / sous-ensemble contrat est faite côté
# ``structurer.py`` après réception. ``enum`` sur une chaîne est supporté en mode
# strict et garantit une des valeurs de ``TypeDocument`` (source unique des valeurs
# autorisées).
_STRUCTURATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **_INVOICE_SCHEMA["properties"],
        "type_document": {
            "type": "string",
            "enum": [type_doc.value for type_doc in TypeDocument],
        },
        "delai_paiement_jours": {"type": ["integer", "null"]},
    },
    "required": [
        *_INVOICE_SCHEMA["required"],
        "type_document",
        "delai_paiement_jours",
    ],
}

# ``response_format`` complet à passer à ``call_llm`` (structured outputs strict).
INVOICE_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": _SCHEMA_NAME,
        "strict": True,
        "schema": _STRUCTURATION_SCHEMA,
    },
}
