"""Fabrique des documents d'exemple utilisés par les tests.

Tous les documents sont **générés en mémoire** à l'exécution (reportlab pour les
PDF, PyMuPDF pour les images) : aucun binaire n'est versionné dans le dépôt.
Deux conséquences volontaires — le dépôt reste léger, et il est structurellement
impossible d'y déposer par accident une vraie facture.

**Aucune donnée réelle.** Les identifiants ci-dessous sont inventés, mais
*valides au sens des contrôles d'intégrité* : le SIRET passe la clé de Luhn et
l'IBAN la clé mod-97. C'est indispensable — avec des identifiants invalides, le
scoring de confiance ne verrait jamais le chemin nominal et les tests
mesureraient toujours une extraction dégradée.

Les dates sont **relatives au jour d'exécution** plutôt que figées : une date
d'émission codée en dur finirait par sortir de la plage de plausibilité de
``confidence.py`` et ferait échouer les tests des années plus tard, sans qu'aucun
comportement n'ait changé.
"""

import io
from datetime import date, timedelta
from decimal import Decimal

import fitz
from reportlab.pdfgen import canvas

# --- Identifiants fictifs ---------------------------------------------------

# SIREN inventés (111222333 / 444555666) complétés d'un NIC choisi pour que le
# SIRET complet passe la clé de Luhn. Aucune entreprise réelle derrière.
SIRET_EMETTEUR = "11122233300010"
SIRET_DESTINATAIRE = "44455566600023"

# IBAN fictif : BBAN volontairement répétitif (11111 22222 33333...), clé de
# contrôle calculée pour satisfaire le mod-97. Aucun compte réel derrière.
IBAN = "FR5411111222223333344444555"

NUMERO_FACTURE = "FA-2026-042"

# --- Montants et dates ------------------------------------------------------

# Facture cohérente de bout en bout : 2 × 500,00 = 1000,00 HT, TVA 20 %,
# HT + TVA = TTC. Le triplet ET la somme des lignes se recoupent, donc le
# scoring emprunte son chemin nominal (tous les contrôles croisés au vert).
QUANTITE = Decimal("2")
PRIX_UNITAIRE_HT = Decimal("500.00")
TAUX_TVA = Decimal("20")
TOTAL_HT = Decimal("1000.00")
TOTAL_TVA = Decimal("200.00")
TOTAL_TTC = Decimal("1200.00")

DESIGNATION = "Prestation de conseil"

DATE_EMISSION = date.today() - timedelta(days=30)
DATE_ECHEANCE = DATE_EMISSION + timedelta(days=60)  # à venir : cas normal


# --- Générateurs génériques -------------------------------------------------


def pdf_with_pages(*pages: str | None) -> bytes:
    """PDF natif dont chaque argument est le texte d'une page.

    ``None`` produit une page **sans couche texte** : c'est ainsi qu'on simule un
    scan (une page valide dont rien n'est extractible), sans embarquer d'image.

    Args:
        pages: texte de chaque page, dans l'ordre. ``None`` pour une page vide.

    Returns:
        Les octets du PDF généré.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        if text is not None:
            pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def pdf_multiligne(lignes: list[str]) -> bytes:
    """PDF natif d'une page portant plusieurs lignes de texte.

    Utilisé pour les documents d'exemple réalistes : une facture tient sur une
    page mais compte une quinzaine de lignes.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    y = 780
    for ligne in lignes:
        pdf.drawString(72, y, ligne)
        y -= 18
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- Documents d'exemple ----------------------------------------------------


def facture_native_pdf() -> bytes:
    """PDF natif d'une facture complète et cohérente (cas nominal).

    Document de référence des tests de composition : toutes les informations du
    contrat y figurent, les montants se recoupent, les identifiants sont valides.
    """
    return pdf_multiligne(
        [
            "SOCIETE EXEMPLE SARL",
            "12 rue de la Demonstration, 75001 Paris",
            f"SIRET : {SIRET_EMETTEUR}",
            "",
            f"FACTURE N {NUMERO_FACTURE}",
            f"Date d'emission : {DATE_EMISSION.strftime('%d/%m/%Y')}",
            f"Date d'echeance : {DATE_ECHEANCE.strftime('%d/%m/%Y')}",
            "",
            "Client : ENTREPRISE FICTIVE SAS",
            f"SIRET : {SIRET_DESTINATAIRE}",
            "",
            "Designation              Qte    PU HT      TVA",
            f"{DESIGNATION}      2      500,00     20 %",
            "",
            "Total HT                        1 000,00 EUR",
            "TVA 20 %                          200,00 EUR",
            "Total TTC                       1 200,00 EUR",
            "",
            f"IBAN : {IBAN}",
        ]
    )


def facture_partielle_pdf() -> bytes:
    """PDF natif d'une facture incomplète : en-tête et totaux, sans détail.

    Cas fréquent en production (facture d'acompte, ticket) : exploitable, mais
    plusieurs champs du contrat manquent — le score doit baisser sans tomber à 0.
    """
    return pdf_multiligne(
        [
            "SOCIETE EXEMPLE SARL",
            f"SIRET : {SIRET_EMETTEUR}",
            "",
            f"FACTURE N {NUMERO_FACTURE}",
            "",
            "Total TTC                       1 200,00 EUR",
        ]
    )


def document_non_facture_pdf() -> bytes:
    """PDF natif lisible mais qui n'est pas une facture (courrier).

    Le texte s'extrait parfaitement : l'échec attendu vient de l'aval (le modèle
    ne trouve aucun champ de facture), pas de la préparation des données.
    """
    return pdf_multiligne(
        [
            "SOCIETE EXEMPLE SARL",
            "",
            "Objet : convocation a l'assemblee generale",
            "",
            "Madame, Monsieur,",
            "Nous avons le plaisir de vous convier a notre assemblee",
            "generale ordinaire qui se tiendra le mois prochain.",
            "",
            "Veuillez agreer nos salutations distinguees.",
        ]
    )


def pdf_scanne(pages: int = 1) -> bytes:
    """PDF « scanné » : ``pages`` pages valides mais sans aucune couche texte.

    C'est exactement ce que voit le détecteur face à un vrai scan : un PDF qui
    s'ouvre sans erreur et dont l'extraction native ne rend rien.
    """
    return pdf_with_pages(*([None] * pages))


def image_facture_png() -> bytes:
    """Vraie image PNG portant le texte d'une facture (rendu PyMuPDF).

    L'OCR est mocké en test, mais l'image, elle, est réelle : elle prouve que le
    chemin image accepte de véritables octets PNG et qu'aucune étape n'essaie de
    les interpréter comme un PDF.
    """
    document = fitz.open()
    page = document.new_page()
    y = 100
    for ligne in (
        f"FACTURE N {NUMERO_FACTURE}",
        f"SIRET : {SIRET_EMETTEUR}",
        "Total TTC 1 200,00 EUR",
    ):
        page.insert_text((72, y), ligne)
        y += 24
    return bytes(page.get_pixmap(dpi=150).tobytes("png"))


# --- Documents dégradés -----------------------------------------------------

# Octets qui commencent comme un PDF mais n'en sont pas un : le parseur échoue à
# l'ouverture. Cas « fichier corrompu / tronqué » des règles de validation.
PDF_CORROMPU = b"%PDF-1.4 ceci n'est pas un vrai PDF"

# Fichier vide : accepté à l'entrée (le type MIME est correct, la taille aussi),
# rejeté en aval faute de contenu exploitable.
FICHIER_VIDE = b""


def octets_de_taille(octets: int) -> bytes:
    """Charge utile de taille exacte, pour les tests de plafond de taille.

    Le contrôle de taille de l'endpoint mesure les octets reçus **avant** toute
    tentative de parsing : le contenu n'a donc pas besoin d'être un PDF valide,
    et générer un vrai PDF de plusieurs mégaoctets ne ferait que ralentir la
    suite sans rien tester de plus.
    """
    entete = b"%PDF-1.4\n"
    return entete + b"\0" * max(0, octets - len(entete))
