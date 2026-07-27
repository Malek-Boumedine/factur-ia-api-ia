"""Tests du calcul de score de confiance (``compute_confidence``).

Heuristiques déterministes pures : aucun mock, aucun appel réseau. On vérifie que
le score par-champ reflète la validité/cohérence des données et que le score global
agrège correctement — sans jamais tomber sur 0 (sentinelle « inexploitable »
réservée à l'orchestrateur).

Fixtures : SIRET et IBAN réellement valides (clé de Luhn / mod-97 vérifiées).
"""

from decimal import Decimal
from typing import Any

from src.extractions.confidence import (
    _CROSS_CHECK_PENALTY,
    _FLOOR,
    _INTEGRITY_FAILED,
    _MALFORMED,
    _UNVERIFIED_PRESENT,
    compute_confidence,
)

# SIRET valides (Luhn ok) et invalide (14 chiffres mais Luhn ko).
_SIRET_VALIDE = "73282932000074"
_SIRET_VALIDE_2 = "55210055400021"
_SIRET_LUHN_KO = "73282932000064"
# IBAN valide (mod-97 ok).
_IBAN_VALIDE = "FR7630006000011234567890189"


def _facture(**overrides: Any) -> dict[str, Any]:
    """Facture nominale entièrement valide et cohérente, surchargeable champ à champ."""
    facture: dict[str, Any] = {
        "siret_emetteur": _SIRET_VALIDE,
        "siret_destinataire": _SIRET_VALIDE_2,
        "numero_facture": "FA-2026-042",
        "date_emission": "2026-01-15",
        "total_ht": Decimal("1000.00"),
        "total_tva": Decimal("200.00"),
        "total_ttc": Decimal("1200.00"),
        "iban": _IBAN_VALIDE,
        "lignes": [
            {
                "designation": "Prestation de conseil",
                "quantite": 2,
                "prix_unitaire_ht": Decimal("500.00"),
                "taux_tva": Decimal("20.00"),
            }
        ],
    }
    facture.update(overrides)
    return facture


# --- Cas nominal -----------------------------------------------------------


def test_cas_nominal_score_eleve() -> None:
    """Facture entièrement valide et cohérente → score global très élevé.

    Plafonné juste sous 1 par le seul ``numero_facture`` (texte libre non
    vérifiable, ``0.7``) : c'est un choix honnête, pas un bug.
    """
    result = compute_confidence(_facture())

    assert Decimal("0.95") <= result.score_global < Decimal("1")
    # Tous les champs vérifiables sont au maximum, sauf le texte libre non vérifiable.
    assert result.par_champ["siret_emetteur"] == Decimal("1")
    assert result.par_champ["iban"] == Decimal("1")
    assert result.par_champ["total_ttc"] == Decimal("1")
    assert result.par_champ["date_emission"] == Decimal("1")
    assert result.par_champ["numero_facture"] == _UNVERIFIED_PRESENT


def test_par_champ_contient_tous_les_champs() -> None:
    """``par_champ`` expose toujours les neuf champs (prêt pour le surlignage front)."""
    result = compute_confidence(_facture())

    assert set(result.par_champ) == {
        "siret_emetteur",
        "siret_destinataire",
        "numero_facture",
        "date_emission",
        "total_ht",
        "total_tva",
        "total_ttc",
        "iban",
        "lignes",
    }


def test_score_toujours_dans_intervalle() -> None:
    """Le score global reste dans ``(0, 1]`` et quantifié à 4 décimales."""
    result = compute_confidence(_facture())
    assert Decimal("0") < result.score_global <= Decimal("1")
    assert result.score_global.as_tuple().exponent == -4


# --- Validité par champ ----------------------------------------------------


def test_siret_luhn_invalide_penalise() -> None:
    """SIRET à 14 chiffres mais clé de Luhn fausse → confiance dégradée, pas nulle."""
    result = compute_confidence(_facture(siret_emetteur=_SIRET_LUHN_KO))
    assert result.par_champ["siret_emetteur"] == _INTEGRITY_FAILED


def test_siret_mal_forme_penalise_davantage() -> None:
    """SIRET de mauvaise longueur → plus pénalisé qu'une simple clé fausse."""
    result = compute_confidence(_facture(siret_emetteur="12345"))
    assert result.par_champ["siret_emetteur"] == _MALFORMED


def test_iban_invalide_penalise() -> None:
    """IBAN présent mais mod-97 faux → confiance dégradée."""
    result = compute_confidence(_facture(iban="FR7630006000011234567890188"))
    assert result.par_champ["iban"] == _INTEGRITY_FAILED


def test_taux_tva_illegal_degrade_les_lignes() -> None:
    """Un taux de TVA hors taux légaux fait baisser la confiance des lignes.

    La ligne somme au total HT du fixture (1 × 1000) : la cohérence croisée est
    bonne, on isole le seul contrôle du taux.
    """
    lignes = [
        {
            "designation": "Article",
            "quantite": 1,
            "prix_unitaire_ht": Decimal("1000.00"),
            "taux_tva": Decimal("17.00"),  # taux inexistant en France
        }
    ]
    result = compute_confidence(_facture(lignes=lignes))
    # 3 contrôles sur 4 passent (désignation, quantité, prix) → 0.75.
    assert result.par_champ["lignes"] == Decimal("0.75")


def test_date_non_parsable_penalise() -> None:
    """Date présente mais non parsable → confiance dégradée (mal formée)."""
    result = compute_confidence(_facture(date_emission="15/01/2026"))
    assert result.par_champ["date_emission"] == _MALFORMED


# --- Champs manquants ------------------------------------------------------


def test_champs_critiques_manquants_font_chuter_le_score() -> None:
    """Totaux et SIRET émetteur absents → score bas mais strictement positif."""
    result = compute_confidence(
        _facture(
            siret_emetteur=None,
            total_ht=None,
            total_tva=None,
            total_ttc=None,
        )
    )
    assert result.par_champ["siret_emetteur"] == Decimal("0")
    assert result.par_champ["total_ht"] == Decimal("0")
    assert result.score_global < Decimal("0.5")
    assert result.score_global > Decimal("0")


def test_champ_optionnel_absent_non_penalisant() -> None:
    """Un IBAN absent (non critique) n'est pas pénalisé comme un IBAN invalide.

    Absent → exclu de la moyenne (score quasi inchangé, marginalement plus bas car
    un champ à confiance 1 sort d'une moyenne < 1). Invalide → compté et pénalisé.
    L'absence doit rester bien mieux notée que l'invalidité.
    """
    sans_iban = compute_confidence(_facture(iban=None)).score_global
    iban_invalide = compute_confidence(
        _facture(iban="FR7630006000011234567890188")
    ).score_global

    assert sans_iban > iban_invalide  # l'absence n'est pas pénalisée
    assert sans_iban > Decimal("0.9")  # et reste élevée
    # Mais le champ reste signalé à 0 pour un éventuel surlignage.
    assert compute_confidence(_facture(iban=None)).par_champ["iban"] == Decimal("0")


# --- Incohérence arithmétique ----------------------------------------------


def test_incoherence_arithmetique_degrade_les_totaux() -> None:
    """HT + TVA ≠ TTC → les trois totaux voient leur confiance baisser."""
    result = compute_confidence(_facture(total_ttc=Decimal("9999.00")))
    assert result.par_champ["total_ht"] == _INTEGRITY_FAILED
    assert result.par_champ["total_tva"] == _INTEGRITY_FAILED
    assert result.par_champ["total_ttc"] == _INTEGRITY_FAILED
    # Le score global reflète l'incohérence : plus bas que le cas cohérent.
    assert result.score_global < compute_confidence(_facture()).score_global


def test_tolerance_centime_sur_les_totaux() -> None:
    """Un écart d'un centime (arrondi) reste considéré comme cohérent."""
    result = compute_confidence(_facture(total_ttc=Decimal("1200.01")))
    assert result.par_champ["total_ttc"] == Decimal("1")


def test_totaux_presents_mais_non_recoupables() -> None:
    """Total présent sans les autres → présent mais non vérifiable (ni 1, ni 0).

    Lignes du fixture cohérentes avec le total HT (2 × 500 = 1000) : le contrôle
    croisé n'interfère pas, on isole le verdict « non recoupable » du triplet.
    """
    result = compute_confidence(_facture(total_tva=None, total_ttc=None))
    assert result.par_champ["total_ht"] == _UNVERIFIED_PRESENT


# --- Cohérence croisée lignes / total HT -------------------------------------


def _lignes(*prix: str) -> list[dict[str, Any]]:
    """Lignes structurellement valides (quantité 1, taux légal) aux prix donnés."""
    return [
        {
            "designation": f"Prestation {i}",
            "quantite": 1,
            "prix_unitaire_ht": Decimal(p),
            "taux_tva": Decimal("20.00"),
        }
        for i, p in enumerate(prix, start=1)
    ]


def test_lignes_coherentes_avec_total_score_inchange() -> None:
    """Somme des lignes = total HT → aucun effet, le score structurel reste plein."""
    result = compute_confidence(_facture())  # 2 × 500 = 1000 = total_ht
    assert result.par_champ["lignes"] == Decimal("1")
    assert result.par_champ["total_ht"] == Decimal("1")


def test_ligne_sous_evaluee_confirmee_fait_chuter_le_score() -> None:
    """Cas réel : ligne extraite 850 au lieu de 1850, triplet cohérent.

    850 + 480 + 660 = 1990 ≠ 2990, et HT + TVA = TTC se corroborent → l'erreur
    est imputée aux lignes (plafond) et le malus global s'applique : le score
    tombe mathématiquement sous ``_CROSS_CHECK_PENALTY``, donc sous le seuil
    d'alerte 0.7 — l'erreur monétaire devient visible sur le seul score transmis.
    """
    result = compute_confidence(
        _facture(
            total_ht=Decimal("2990.00"),
            total_tva=Decimal("598.00"),
            total_ttc=Decimal("3588.00"),
            lignes=_lignes("850.00", "480.00", "660.00"),
        )
    )

    assert result.par_champ["lignes"] == _INTEGRITY_FAILED
    # Triplet cohérent : les totaux gardent leur confiance pleine.
    assert result.par_champ["total_ht"] == Decimal("1")
    assert result.score_global <= _CROSS_CHECK_PENALTY
    assert result.score_global < Decimal("0.7")


def test_incoherence_non_confirmee_plafonds_sans_malus() -> None:
    """Triplet invérifiable : coupable indésignable → lignes ET total HT plafonnés.

    Pas de malus global (le signal est ambigu, un total absent peut expliquer
    l'écart) : la pénalité passe uniquement par les plafonds par champ.
    """
    result = compute_confidence(
        _facture(
            total_ht=Decimal("2990.00"),
            total_tva=None,
            total_ttc=None,
            lignes=_lignes("850.00", "480.00", "660.00"),
        )
    )

    assert result.par_champ["lignes"] == _INTEGRITY_FAILED
    assert result.par_champ["total_ht"] == _INTEGRITY_FAILED


def test_pas_de_lignes_verdict_neutre() -> None:
    """Sans lignes, le contrôle croisé est invérifiable : aucun plafond appliqué."""
    result = compute_confidence(_facture(lignes=[], total_tva=None, total_ttc=None))
    # Sans le verdict neutre, total_ht serait plafonné à _INTEGRITY_FAILED.
    assert result.par_champ["total_ht"] == _UNVERIFIED_PRESENT


def test_total_ht_absent_verdict_neutre() -> None:
    """Sans total HT, le contrôle croisé est invérifiable : lignes non pénalisées."""
    result = compute_confidence(_facture(total_ht=None))
    assert result.par_champ["lignes"] == Decimal("1")


def test_ligne_au_prix_null_verdict_neutre() -> None:
    """Une ligne sans prix rend la somme incalculable : pas de fausse incohérence.

    La ligne reste pénalisée structurellement (prix manquant → 3 contrôles sur 4),
    mais ni plafond croisé ni malus — et le total HT garde sa confiance.
    """
    lignes = _lignes("1000.00")
    lignes[0]["prix_unitaire_ht"] = None
    result = compute_confidence(_facture(lignes=lignes))

    assert result.par_champ["lignes"] == Decimal("0.75")
    assert result.par_champ["total_ht"] == Decimal("1")


def test_tolerance_proportionnelle_au_nombre_de_lignes() -> None:
    """L'arrondi au centime par ligne est absorbé (tolérance × nombre de lignes)."""
    # 3 × 333.33 = 999.99, écart de 0.01 sur total_ht 1000 → cohérent.
    result = compute_confidence(_facture(lignes=_lignes("333.33", "333.33", "333.33")))
    assert result.par_champ["lignes"] == Decimal("1")


def test_malus_ne_produit_jamais_zero() -> None:
    """Extraction catastrophique + malus confirmé → score plancher, jamais 0.

    Le 0 reste le sentinelle de l'orchestrateur : le malus, multiplicatif et
    appliqué avant ``max(raw, _FLOOR)``, ne peut pas le produire.
    """
    lignes = [
        {
            "designation": "",
            "quantite": 1,
            "prix_unitaire_ht": Decimal("999.00"),
            "taux_tva": Decimal("17.00"),
        }
    ]
    result = compute_confidence(
        _facture(
            siret_emetteur=None,
            siret_destinataire=None,
            numero_facture=None,
            date_emission=None,
            iban=None,
            total_ht=Decimal("100.00"),
            total_tva=Decimal("20.00"),
            total_ttc=Decimal("120.00"),  # triplet cohérent → malus confirmé
            lignes=lignes,
        )
    )

    assert result.score_global >= _FLOOR
    assert result.score_global > Decimal("0")
    assert result.score_global <= _CROSS_CHECK_PENALTY


# --- Extraction inexploitable (jamais 0) -----------------------------------


def test_extraction_vide_ne_tombe_jamais_a_zero() -> None:
    """Extraction parsée mais entièrement vide → score plancher, jamais 0.

    Le 0 est un sentinelle réservé à l'orchestrateur (échec du pipeline). Une
    extraction parsée, même catastrophique, reste strictement positive.
    """
    facture_vide: dict[str, Any] = {
        "siret_emetteur": None,
        "siret_destinataire": None,
        "numero_facture": None,
        "date_emission": None,
        "total_ht": None,
        "total_tva": None,
        "total_ttc": None,
        "iban": None,
        "lignes": [],
    }
    result = compute_confidence(facture_vide)

    assert result.score_global == _FLOOR
    assert result.score_global > Decimal("0")
    # Tous les champs sont signalés à 0 → tout serait surligné côté front.
    assert all(conf == Decimal("0") for conf in result.par_champ.values())


def test_facture_partielle_reste_positive() -> None:
    """Même une extraction très pauvre garde un score strictement positif."""
    result = compute_confidence(
        _facture(
            siret_emetteur=None,
            siret_destinataire=None,
            iban=None,
            numero_facture=None,
            date_emission=None,
            total_ht=None,
            total_tva=None,
            total_ttc=None,
        )
    )
    assert result.score_global > Decimal("0")


# --- Robustesse des entrées ------------------------------------------------


def test_date_future_implausible_degradee() -> None:
    """Date parsable mais hors plage plausible (futur lointain) → confiance moyenne."""
    result = compute_confidence(_facture(date_emission="2999-12-31"))
    assert result.par_champ["date_emission"] == Decimal("0.5")


def test_ligne_non_dict_ignoree() -> None:
    """Une entrée de ligne qui n'est pas un objet → confiance nulle pour cette ligne."""
    result = compute_confidence(_facture(lignes=["texte parasite"]))
    assert result.par_champ["lignes"] == Decimal("0")


def test_montants_en_chaine_convertis() -> None:
    """Un total transmis en chaîne numérique reste exploité (conversion Decimal)."""
    result = compute_confidence(
        _facture(total_ht="1000.00", total_tva="200.00", total_ttc="1200.00")
    )
    assert result.par_champ["total_ttc"] == Decimal("1")


def test_iban_caractere_invalide_penalise() -> None:
    """IBAN contenant un caractère non alphanumérique → mod-97 impossible, dégradé."""
    result = compute_confidence(_facture(iban="FR76-3000-6000-011@"))
    assert result.par_champ["iban"] == _INTEGRITY_FAILED


def test_total_chaine_non_numerique_traite_comme_absent() -> None:
    """Un total en chaîne non convertible → traité comme absent (0), jamais d'erreur."""
    result = compute_confidence(_facture(total_ht="illisible"))
    assert result.par_champ["total_ht"] == Decimal("0")
