"""Réponses simulées du modèle Groq, pour la structuration mockée.

Le LLM n'est **jamais** appelé en test : ces fabriques produisent la chaîne que
``call_llm`` est censé renvoyer, et le pipeline la traite comme si elle venait du
modèle. Tout ce qui est en aval (parsing, dérivation d'échéance, scoring,
validation contrat) s'exécute donc pour de vrai.

Les réponses sont des chaînes JSON — pas des ``dict`` — parce que c'est bien une
chaîne que renvoie le SDK Groq : c'est le parsing lui-même qu'on veut exercer,
avec ses cas de bord (JSON tronqué, valeur inattendue, champ hors schéma).

Miroir du schéma LLM **à plat** (cf. ``src/extractions/prompts.py``) : les dix
champs du contrat, plus les deux champs hors contrat ``type_document`` et
``delai_paiement_jours``.
"""

import json
from typing import Any

from tests.fixtures import documents

# Réponse tronquée en plein objet : ce que produit une limite de tokens atteinte.
JSON_TRONQUE = '{"numero_facture": "FA-2026-042", "total_ht": 1000.0'

# Réponse vide : le modèle n'a rien renvoyé (contenu absent).
REPONSE_VIDE = ""

# JSON syntaxiquement valide mais qui n'est pas un objet : le schéma strict
# l'interdit, un modèle en dérive peut malgré tout le produire.
JSON_NON_OBJET = '["FA-2026-042", 1000.0]'


def reponse_facture(**overrides: Any) -> str:
    """Réponse modèle d'une facture complète et cohérente, surchargeable.

    Les valeurs reflètent le document ``documents.facture_native_pdf()`` : c'est
    ce qu'un modèle correct devrait en tirer. Les montants sont des ``float``
    JSON (comme dans une vraie réponse) — c'est le pipeline qui les convertit en
    ``Decimal``.

    Args:
        overrides: champs à remplacer ou à ajouter. Une valeur ``None`` produit
            un ``null`` JSON ; une clé inconnue du schéma simule un modèle qui
            invente un champ.

    Returns:
        La réponse du modèle, sérialisée en JSON.
    """
    payload: dict[str, Any] = {
        "siret_emetteur": documents.SIRET_EMETTEUR,
        "siret_destinataire": documents.SIRET_DESTINATAIRE,
        "numero_facture": documents.NUMERO_FACTURE,
        "date_emission": documents.DATE_EMISSION.isoformat(),
        "date_echeance": documents.DATE_ECHEANCE.isoformat(),
        "total_ht": 1000.00,
        "total_tva": 200.00,
        "total_ttc": 1200.00,
        "iban": documents.IBAN,
        "lignes": [
            {
                "designation": documents.DESIGNATION,
                "quantite": 2,
                "prix_unitaire_ht": 500.00,
                "taux_tva": 20.00,
            }
        ],
        "type_document": "facture",
        "delai_paiement_jours": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def reponse_partielle() -> str:
    """Réponse modèle d'une facture incomplète : en-tête et TTC seulement.

    Exploitable (un total est présent), mais la moitié des champs manque : le
    score doit refléter cette dégradation sans jamais atteindre le sentinelle 0.
    """
    return reponse_facture(
        siret_destinataire=None,
        date_emission=None,
        date_echeance=None,
        total_ht=None,
        total_tva=None,
        iban=None,
        lignes=[],
    )


def reponse_vide_de_sens() -> str:
    """Réponse modèle sur un document qui n'est pas une facture.

    Le modèle a bien répondu et respecté le schéma, mais n'a rien trouvé à
    extraire : aucun total, aucune ligne. C'est le cas « inexploitable » —
    il n'y a rien qu'un humain puisse corriger.
    """
    return reponse_facture(
        siret_emetteur=None,
        siret_destinataire=None,
        numero_facture=None,
        date_emission=None,
        date_echeance=None,
        total_ht=None,
        total_tva=None,
        total_ttc=None,
        iban=None,
        lignes=[],
        type_document="inconnu",
    )
