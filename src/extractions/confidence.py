"""Calcul du score de confiance d'une extraction de facture.

Dernière brique métier avant le callback : à partir du sous-ensemble « données
extraites » (``facture``) produit par ``structurer.py``, on estime **à quel point
faire confiance** à chaque champ, puis on agrège en un score global.

Approche retenue : **heuristiques déterministes pures**, sans appel LLM. On mesure
la *validité* et la *cohérence interne* des champs (clé de Luhn du SIRET, mod-97 de
l'IBAN, égalité ``HT + TVA = TTC``, taux de TVA légaux, date ISO plausible), pas
leur *fidélité à la source* — un SIRET bien formé peut rester le mauvais numéro.
C'est le plafond assumé de l'approche, et la raison d'être de la relecture humaine
(human-in-the-loop). Avantage : objectif, reproductible, testable, explicable — un
score qui a un sens réel, pas décoratif. Une auto-évaluation par le LLM serait
complaisante ; elle pourra être ajoutée plus tard en appoint plafonné.

Deux sorties dans ``ConfidenceResult`` :

- ``score_global`` (``Decimal`` dans ``(0, 1]``) : le seul transmis au callback, il
  alimente ``OcrWebhookPayload.score_confiance``.
- ``par_champ`` (``dict`` champ → confiance) : **interne, non transmis**. Le contrat
  ne porte qu'un score global ; ces confiances par-champ sont calculées et prêtes
  pour un futur surlignage champ par champ côté front, quand le contrat sera étendu
  (tâche cross-service séparée). Voir ``src/callback/schemas.py``.

Marqueur d'échec : ``score_confiance = 0`` est un **sentinelle réservé** signifiant
« extraction inexploitable », émis *uniquement* par l'orchestrateur sur le chemin
d'exception (``LlmClientError`` / ``LlmStructurationError``). Ce module ne s'exécute
que sur une extraction déjà parsée et **ne renvoie jamais 0** : un plancher
strictement positif (``_FLOOR``) garantit qu'une extraction valide reste ``> 0``,
même catastrophique. Aucune collision possible avec le sentinelle.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

# --- Constantes de score ---------------------------------------------------

# Plancher strictement positif du score global : garantit qu'une extraction parsée
# ne tombe jamais sur 0 (sentinelle « inexploitable » réservée à l'orchestrateur).
_FLOOR = Decimal("0.05")

# Confiance d'un champ présent mais **non vérifiable** (texte libre : numéro de
# facture, ou total présent qu'on ne peut pas recouper faute des autres). On ne peut
# ni confirmer ni infirmer sa justesse → confiance moyenne, ni pleine ni nulle.
_UNVERIFIED_PRESENT = Decimal("0.7")

# Confiance d'un champ présent, bien dimensionné, mais dont le contrôle d'intégrité
# échoue (SIRET 14 chiffres qui ne passe pas Luhn, totaux HT+TVA≠TTC) : forme
# plausible mais valeur douteuse, typiquement une erreur OCR.
_INTEGRITY_FAILED = Decimal("0.4")

# Confiance d'un champ présent mais manifestement mal formé (mauvaise longueur).
_MALFORMED = Decimal("0.2")

# Précision de restitution du score global.
_QUANTUM = Decimal("0.0001")

# --- Constantes de validation ----------------------------------------------

# Tolérance d'égalité des montants (en euros) : absorbe les arrondis au centime.
_AMOUNT_TOLERANCE = Decimal("0.02")

# Taux de TVA légaux en France (en pourcentage). Un taux hors de cet ensemble est
# suspect. Comparaison numérique (``Decimal("20.00") == Decimal("20")``).
_LEGAL_VAT_RATES = frozenset(
    {Decimal("0"), Decimal("2.1"), Decimal("5.5"), Decimal("10"), Decimal("20")}
)

# Borne basse de plausibilité d'une date d'émission (bornage haut = aujourd'hui).
_MIN_PLAUSIBLE_DATE = date(2000, 1, 1)

# --- Pondération et criticité des champs -----------------------------------

# Poids relatifs dans la moyenne pondérée du score global. Les champs critiques
# (totaux, SIRET émetteur) pèsent le plus.
_FIELD_WEIGHTS: dict[str, Decimal] = {
    "total_ht": Decimal("3"),
    "total_tva": Decimal("2"),
    "total_ttc": Decimal("3"),
    "siret_emetteur": Decimal("3"),
    "siret_destinataire": Decimal("1"),
    "numero_facture": Decimal("2"),
    "date_emission": Decimal("2"),
    "iban": Decimal("1"),
    "lignes": Decimal("2"),
}

# Champs critiques : leur absence est *comptée* (confiance 0 tirant le score vers le
# bas). Les champs non critiques absents sont *exclus* de la moyenne (une facture
# sans IBAN n'est pas pénalisée), mais restent signalés à 0 dans ``par_champ``.
_CRITICAL_FIELDS = frozenset({"total_ht", "total_tva", "total_ttc", "siret_emetteur"})


@dataclass(frozen=True)
class ConfidenceResult:
    """Résultat du calcul de confiance d'une extraction.

    Attributes:
        score_global: confiance globale dans ``(0, 1]`` (jamais 0), destinée à
            ``OcrWebhookPayload.score_confiance`` — le seul champ transmis.
        par_champ: confiance par champ (``0`` à ``1``), **interne, non transmise**.
            Prête pour un futur surlignage champ par champ côté front.
    """

    score_global: Decimal
    par_champ: dict[str, Decimal]


# --- Helpers de bas niveau (purs) ------------------------------------------


def _as_decimal(value: Any) -> Decimal | None:
    """Convertit une valeur numérique en ``Decimal``, ou ``None`` si impossible.

    Le texte extrait est parsé avec ``parse_float=Decimal`` : les montants sont déjà
    des ``Decimal``, mais un entier JSON (ex. ``quantite: 1``) reste un ``int``. On
    couvre ``Decimal``, ``int`` et, par prudence, une chaîne numérique.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    return None


def _luhn_ok(digits: str) -> bool:
    """Vérifie la clé de Luhn d'une suite de chiffres (SIREN/SIRET)."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _iban_mod97_ok(iban: str) -> bool:
    """Vérifie la clé de contrôle mod-97 d'un IBAN (norme ISO 13616)."""
    compact = iban.replace(" ", "").upper()
    rearranged = compact[4:] + compact[:4]
    try:
        numeric = "".join(str(int(char, 36)) for char in rearranged)
    except ValueError:
        return False  # caractère non alphanumérique
    return int(numeric) % 97 == 1


# --- Scoring par champ -----------------------------------------------------


def _score_siret(value: Any) -> Decimal:
    """Confiance d'un SIRET : 14 chiffres + clé de Luhn."""
    if not isinstance(value, str) or not value.strip():
        return Decimal("0")
    compact = value.replace(" ", "")
    if not compact.isdigit() or len(compact) != 14:
        return _MALFORMED
    return Decimal("1") if _luhn_ok(compact) else _INTEGRITY_FAILED


def _score_iban(value: Any) -> Decimal:
    """Confiance d'un IBAN : mod-97 (norme). Vide/absent → 0 (signalé pour le front)."""
    if not isinstance(value, str) or not value.strip():
        return Decimal("0")
    return Decimal("1") if _iban_mod97_ok(value) else _INTEGRITY_FAILED


def _score_date(value: Any) -> Decimal:
    """Confiance d'une date d'émission : ISO parsable + plausible."""
    if not isinstance(value, str) or not value.strip():
        return Decimal("0")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return _MALFORMED  # présente mais non parsable
    if _MIN_PLAUSIBLE_DATE <= parsed <= date.today():
        return Decimal("1")
    return Decimal("0.5")  # parsable mais hors plage plausible (futur, trop ancienne)


def _score_text(value: Any) -> Decimal:
    """Confiance d'un champ texte libre non vérifiable (ex. numéro de facture)."""
    if not isinstance(value, str) or not value.strip():
        return Decimal("0")
    return _UNVERIFIED_PRESENT


def _coherence(
    ht: Decimal | None, tva: Decimal | None, ttc: Decimal | None
) -> bool | None:
    """Cohérence arithmétique ``HT + TVA = TTC`` (tolérance centime).

    Renvoie ``None`` si le contrôle est impossible (au moins un total absent).
    """
    if ht is None or tva is None or ttc is None:
        return None
    return abs((ht + tva) - ttc) <= _AMOUNT_TOLERANCE


def _score_total(value: Decimal | None, coherent: bool | None) -> Decimal:
    """Confiance d'un total, modulée par la cohérence croisée des trois totaux.

    La cohérence est une propriété du triplet (elle ne dit pas *lequel* est faux) :
    en cas d'incohérence, les trois totaux voient leur confiance baisser — c'est le
    signal correct pour attirer la relecture humaine sur l'ensemble des montants.
    """
    if value is None:
        return Decimal("0")
    if coherent is True:
        return Decimal("1")
    if coherent is False:
        return _INTEGRITY_FAILED
    return _UNVERIFIED_PRESENT  # présent mais non recoupable (autres totaux absents)


def _score_lignes(value: Any) -> Decimal:
    """Confiance des lignes : moyenne de la validité structurelle de chaque ligne.

    Par ligne, quatre contrôles : désignation non vide, quantité > 0, prix unitaire
    présent et ≥ 0, taux de TVA parmi les taux légaux. Liste vide/absente → 0.
    """
    if not isinstance(value, list) or not value:
        return Decimal("0")

    line_scores: list[Decimal] = []
    for line in value:
        if not isinstance(line, dict):
            line_scores.append(Decimal("0"))
            continue
        designation = line.get("designation")
        quantite = _as_decimal(line.get("quantite"))
        prix = _as_decimal(line.get("prix_unitaire_ht"))
        taux = _as_decimal(line.get("taux_tva"))
        checks = [
            isinstance(designation, str) and bool(designation.strip()),
            quantite is not None and quantite > 0,
            prix is not None and prix >= 0,
            taux is not None and taux in _LEGAL_VAT_RATES,
        ]
        line_scores.append(Decimal(sum(checks)) / Decimal(len(checks)))

    return sum(line_scores, Decimal("0")) / Decimal(len(line_scores))


# --- Agrégation ------------------------------------------------------------


def _is_absent(field: str, facture: dict[str, Any]) -> bool:
    """Indique si un champ est absent/vide (pour l'exclusion des non-critiques)."""
    value = facture.get(field)
    if value is None:
        return True
    if field == "lignes":
        return not (isinstance(value, list) and value)
    if isinstance(value, str):
        return not value.strip()
    return False


def compute_confidence(facture: dict[str, Any]) -> ConfidenceResult:
    """Calcule la confiance d'une extraction de facture (par champ + globale).

    Applique les heuristiques déterministes à chaque champ du sous-ensemble contrat
    ``facture`` (issu de ``structurer.py``, montants en ``Decimal``), puis agrège en
    une moyenne pondérée : les champs critiques absents sont comptés (confiance 0),
    les non-critiques absents sont exclus de la moyenne mais restent signalés à 0
    dans ``par_champ``. Le résultat est plafonné par le bas à ``_FLOOR`` : il ne vaut
    **jamais 0** (sentinelle « inexploitable » réservée à l'orchestrateur).

    Args:
        facture: sous-ensemble « données extraites » (miroir d'``OcrWebhookPayload``
            sans ``id_document`` ni ``score_confiance``), montants en ``Decimal``.

    Returns:
        ``ConfidenceResult`` : ``score_global`` dans ``(0, 1]`` (transmis au callback)
        et ``par_champ`` (interne, prêt pour le surlignage front futur).
    """
    ht = _as_decimal(facture.get("total_ht"))
    tva = _as_decimal(facture.get("total_tva"))
    ttc = _as_decimal(facture.get("total_ttc"))
    coherent = _coherence(ht, tva, ttc)

    par_champ: dict[str, Decimal] = {
        "siret_emetteur": _score_siret(facture.get("siret_emetteur")),
        "siret_destinataire": _score_siret(facture.get("siret_destinataire")),
        "numero_facture": _score_text(facture.get("numero_facture")),
        "date_emission": _score_date(facture.get("date_emission")),
        "total_ht": _score_total(ht, coherent),
        "total_tva": _score_total(tva, coherent),
        "total_ttc": _score_total(ttc, coherent),
        "iban": _score_iban(facture.get("iban")),
        "lignes": _score_lignes(facture.get("lignes")),
    }

    weighted_sum = Decimal("0")
    weight_total = Decimal("0")
    for field, weight in _FIELD_WEIGHTS.items():
        if _is_absent(field, facture) and field not in _CRITICAL_FIELDS:
            continue  # non-critique absent : ni récompensé ni pénalisé
        weighted_sum += weight * par_champ[field]
        weight_total += weight

    raw = weighted_sum / weight_total if weight_total > 0 else _FLOOR
    score_global = max(raw, _FLOOR).quantize(_QUANTUM)

    return ConfidenceResult(score_global=score_global, par_champ=par_champ)
