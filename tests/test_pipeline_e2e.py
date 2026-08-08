"""Tests de composition : le pipeline complet, sur documents d'exemple.

Les autres fichiers de tests vérifient chaque brique isolément. Celui-ci vérifie
qu'**elles tiennent ensemble** : on poste un vrai document sur l'endpoint et on
inspecte le payload qui part au callback, en ayant traversé pour de bon
l'authentification, la validation d'entrée, la détection PDF, l'extraction de
texte, le parsing de la structuration, le scoring de confiance et la validation
contrat. Un test unitaire vert sur chaque maillon ne prouve pas que la chaîne
tient — c'est ce que ces tests-ci apportent.

**Trois frontières seulement sont simulées**, et ce sont exactement les trois qui
sortent du processus :

- ``structurer.call_llm`` — le modèle Groq. Jamais appelé (coût, non-déterminisme,
  réseau) ; on lui substitue des réponses fixées (``tests/fixtures/llm.py``).
- ``ocr_extractor._get_reader`` — EasyOCR. Jamais appelé (téléchargement de
  modèles de plusieurs centaines de Mo, temps de calcul) ; la *logique* OCR
  (rendu PyMuPDF, découpage par page, concaténation) s'exécute pour de vrai.
- ``service.send_callback`` — le POST vers l'API data. Remplacé par une capture
  des payloads, qui devient le point d'observation des tests.

Tout le reste est le code de production, exécuté tel quel. Les documents sont
générés à la volée (``tests/fixtures/documents.py``) : aucun binaire versionné,
aucune donnée réelle.
"""

import json
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from src.callback.schemas import OcrWebhookPayload
from src.core.config import settings
from src.extractions import ocr_extractor, service, structurer
from src.extractions.llm_client import LlmClientError
from src.main import app

from tests.fixtures import documents
from tests.fixtures import llm as llm_fixtures

client = TestClient(app)

_ENTETES = {"X-OCR-Secret-Token": settings.SECRET_OCR_TOKEN}
_PDF_MIME = "application/pdf"
_PNG_MIME = "image/png"


# --- Doublures des trois frontières ----------------------------------------


class _ModeleSimule:
    """Faux ``call_llm`` : renvoie une réponse fixée et retient ce qu'on lui soumet.

    Retenir ``user_content`` est ce qui permet de vérifier que le texte
    réellement extrait du document est bien celui qui part au modèle — la preuve
    que le maillon extraction et le maillon structuration sont raccordés.
    """

    def __init__(self) -> None:
        self.reponse: str = llm_fixtures.reponse_facture()
        self.erreur: Exception | None = None
        self.textes_soumis: list[str] = []
        self.formats_demandes: list[Any] = []

    def __call__(
        self,
        system_prompt: str,
        user_content: str,
        *,
        response_format: Any = None,
    ) -> str:
        self.textes_soumis.append(user_content)
        self.formats_demandes.append(response_format)
        if self.erreur is not None:
            raise self.erreur
        return self.reponse


class _OcrSimule:
    """Faux ``Reader`` EasyOCR : renvoie des fragments fixés, compte les appels."""

    def __init__(self) -> None:
        self.fragments: list[str] = [
            f"FACTURE N {documents.NUMERO_FACTURE}",
            f"SIRET : {documents.SIRET_EMETTEUR}",
            "Total TTC 1 200,00 EUR",
        ]
        self.appels = 0

    def readtext(self, image: bytes, detail: int = 1) -> list[str]:
        self.appels += 1
        return list(self.fragments)


@pytest.fixture(autouse=True)
def modele(monkeypatch: pytest.MonkeyPatch) -> _ModeleSimule:
    """Substitue le modèle Groq. Autouse : aucun test ne peut l'appeler par oubli."""
    simule = _ModeleSimule()
    monkeypatch.setattr(structurer, "call_llm", simule)
    return simule


@pytest.fixture(autouse=True)
def ocr(monkeypatch: pytest.MonkeyPatch) -> _OcrSimule:
    """Substitue le Reader EasyOCR. Autouse : aucun modèle n'est jamais téléchargé."""
    simule = _OcrSimule()
    monkeypatch.setattr(ocr_extractor, "_get_reader", lambda: simule)
    return simule


@pytest.fixture(autouse=True)
def callbacks(monkeypatch: pytest.MonkeyPatch) -> list[OcrWebhookPayload]:
    """Capture les payloads au lieu de les POSTer : le point d'observation."""
    envoyes: list[OcrWebhookPayload] = []
    monkeypatch.setattr(service, "send_callback", envoyes.append)
    return envoyes


# --- Utilitaires ------------------------------------------------------------


def _poster(
    contenu: bytes,
    *,
    content_type: str = _PDF_MIME,
    id_document: int = 42,
    nom: str = "facture.pdf",
) -> int:
    """Poste un document sur l'endpoint et renvoie le code HTTP.

    Le ``TestClient`` exécute la tâche de fond avant de rendre la main : au
    retour, le pipeline a tourné et le payload est déjà capturé.
    """
    reponse = client.post(
        "/extractions",
        headers=_ENTETES,
        files={"file": (nom, contenu, content_type)},
        data={"id_document": str(id_document)},
    )
    return int(reponse.status_code)


# --- Chemin nominal : PDF natif --------------------------------------------


def test_facture_native_produit_un_payload_conforme(
    callbacks: list[OcrWebhookPayload],
) -> None:
    """Une facture PDF native traverse tout le pipeline et produit le contrat.

    Cas de référence : détection native réelle, extraction pdfplumber réelle,
    scoring et validation réels. Le payload doit porter tous les champs du
    contrat, avec les bons types (``Decimal`` pour les montants, ``date`` pour
    les dates) et un score strictement positif.
    """
    assert _poster(documents.facture_native_pdf()) == 202

    assert len(callbacks) == 1
    payload = callbacks[0]
    assert payload.id_document == 42
    assert payload.score_confiance > 0  # jamais le sentinelle d'échec
    assert payload.numero_facture == documents.NUMERO_FACTURE
    assert payload.siret_emetteur == documents.SIRET_EMETTEUR
    assert payload.siret_destinataire == documents.SIRET_DESTINATAIRE
    assert payload.iban == documents.IBAN
    assert payload.date_emission == documents.DATE_EMISSION
    assert payload.date_echeance == documents.DATE_ECHEANCE
    assert payload.total_ht == Decimal("1000.00")
    assert payload.total_tva == Decimal("200.00")
    assert payload.total_ttc == Decimal("1200.00")
    assert len(payload.lignes) == 1
    assert payload.lignes[0].designation == documents.DESIGNATION


def test_le_texte_extrait_du_pdf_est_bien_celui_soumis_au_modele(
    modele: _ModeleSimule,
) -> None:
    """Le maillon extraction alimente réellement le maillon structuration.

    C'est la vérification qui distingue une chaîne câblée d'une suite de briques
    vertes chacune de son côté : le texte que reçoit le modèle doit être celui
    que ``pdfplumber`` a tiré du document, pas une constante ni un reliquat.
    """
    assert _poster(documents.facture_native_pdf()) == 202

    assert len(modele.textes_soumis) == 1
    soumis = modele.textes_soumis[0]
    # Contenu propre au document généré : il ne peut venir que de l'extraction.
    assert documents.NUMERO_FACTURE in soumis
    assert documents.SIRET_EMETTEUR in soumis
    assert documents.IBAN in soumis
    assert "Total TTC" in soumis
    # Et le schéma strict est bien exigé du modèle au passage.
    assert modele.formats_demandes[0] is not None


def test_type_document_suggere_remonte_au_callback(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """La suggestion de type traverse le pipeline sans être réinterprétée."""
    modele.reponse = llm_fixtures.reponse_facture(type_document="devis")

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].type_document == "devis"


def test_confiance_par_champ_couvre_les_dix_champs(
    callbacks: list[OcrWebhookPayload],
) -> None:
    """Le détail par champ arrive complet au callback (surlignage côté front)."""
    assert _poster(documents.facture_native_pdf()) == 202

    par_champ = callbacks[0].par_champ
    assert par_champ is not None
    assert set(par_champ) == {
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
    }


# --- Chemins d'extraction : scanné et image --------------------------------


def test_pdf_scanne_passe_par_l_ocr(
    ocr: _OcrSimule, modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un PDF sans couche texte est détecté scanné et routé vers l'OCR.

    La détection et le rendu des pages en images (PyMuPDF) sont réels : seul le
    moteur de reconnaissance est simulé. Deux pages doivent donner deux appels.
    """
    assert _poster(documents.pdf_scanne(pages=2)) == 202

    assert ocr.appels == 2  # une reconnaissance par page rendue
    # Le modèle reçoit le texte reconnu par l'OCR, pas du texte de PDF natif.
    assert documents.NUMERO_FACTURE in modele.textes_soumis[0]
    assert callbacks[0].score_confiance > 0


def test_image_png_va_directement_a_l_ocr(
    ocr: _OcrSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Une image PNG saute le détecteur PDF et part directement à l'OCR.

    L'image est un vrai PNG : rien dans la chaîne ne doit tenter de l'ouvrir
    comme un PDF.
    """
    assert (
        _poster(
            documents.image_facture_png(),
            content_type=_PNG_MIME,
            nom="facture.png",
        )
        == 202
    )

    assert ocr.appels == 1
    assert callbacks[0].score_confiance > 0


# --- Évaluation : extractions dégradées mais exploitables ------------------


def test_facture_partielle_score_degrade_mais_exploitable(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Une facture incomplète reste exploitable, avec un score qui le dit.

    Le document a un total mais ni lignes, ni dates, ni IBAN : il y a matière à
    correction humaine, donc pas d'échec — mais le score et le détail par champ
    doivent signaler les manques.
    """
    modele.reponse = llm_fixtures.reponse_partielle()

    assert _poster(documents.facture_partielle_pdf()) == 202

    payload = callbacks[0]
    assert payload.score_confiance > 0  # exploitable : pas le sentinelle
    assert payload.score_confiance < Decimal("0.7")  # mais nettement dégradé
    assert payload.total_ttc == Decimal("1200.00")
    assert payload.total_ht == Decimal("0")  # total absent ramené à 0 par le contrat
    assert payload.par_champ is not None
    assert payload.par_champ["iban"] == Decimal("0")  # absent, signalé au front
    assert payload.par_champ["lignes"] == Decimal("0")


def test_totaux_incoherents_degradent_les_totaux_seulement(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """``HT + TVA ≠ TTC`` fait chuter les trois totaux, sans toucher au reste.

    La dégradation doit être *ciblée* : c'est ce qui permet à l'humain de savoir
    où regarder. Un SIRET valide reste pleinement fiable même si les montants
    ne se recoupent pas.
    """
    modele.reponse = llm_fixtures.reponse_facture(total_ttc=1500.00)

    assert _poster(documents.facture_native_pdf()) == 202

    par_champ = callbacks[0].par_champ
    assert par_champ is not None
    assert par_champ["total_ht"] == Decimal("0.4000")
    assert par_champ["total_tva"] == Decimal("0.4000")
    assert par_champ["total_ttc"] == Decimal("0.4000")
    assert par_champ["siret_emetteur"] == Decimal("1.0000")  # non contaminé


def test_ligne_incoherente_confirmee_passe_sous_le_seuil_d_alerte(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Une erreur monétaire prouvée doit être visible sur le seul score global.

    Les trois totaux se recoupent entre eux, mais la somme des lignes ne tombe
    pas sur le total HT : le coupable est désigné (un montant de ligne est faux).
    Le score doit alors descendre sous le seuil d'alerte du monitoring, même si
    tout le reste de l'extraction est parfait.
    """
    modele.reponse = llm_fixtures.reponse_facture(
        lignes=[
            {
                "designation": documents.DESIGNATION,
                "quantite": 2,
                "prix_unitaire_ht": 400.00,  # 2 × 400 = 800 ≠ 1000 HT
                "taux_tva": 20.00,
            }
        ]
    )

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].score_confiance < settings.MONITORING_SEUIL_ALERTE
    assert callbacks[0].score_confiance > 0


# --- Cas dégradés : documents inexploitables -------------------------------


def test_pdf_corrompu_produit_le_payload_d_echec(
    callbacks: list[OcrWebhookPayload],
) -> None:
    """Un fichier corrompu est accepté à l'entrée puis rejeté par la détection.

    L'endpoint ne peut pas savoir qu'un PDF est illisible sans l'ouvrir : il
    répond 202, et c'est le pipeline qui émet le verdict d'échec.
    """
    assert _poster(documents.PDF_CORROMPU, id_document=55) == 202

    payload = callbacks[0]
    assert payload.id_document == 55
    assert payload.score_confiance == Decimal("0")  # marqueur unique d'échec
    assert payload.lignes == []
    assert payload.par_champ is None
    assert payload.type_document is None


def test_fichier_vide_produit_le_payload_d_echec(
    callbacks: list[OcrWebhookPayload],
) -> None:
    """Un fichier de 0 octet passe la validation d'entrée et échoue en aval."""
    assert _poster(documents.FICHIER_VIDE) == 202

    assert callbacks[0].score_confiance == Decimal("0")


def test_page_blanche_produit_le_payload_d_echec(
    ocr: _OcrSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un PDF sans texte dont l'OCR ne reconnaît rien : scan raté, page blanche.

    Chemin complet et réaliste : détection ``scanned`` → rendu des pages → OCR
    qui ne rend aucun fragment → extraction inexploitable.
    """
    ocr.fragments = []

    assert _poster(documents.pdf_scanne()) == 202

    assert ocr.appels == 1
    assert callbacks[0].score_confiance == Decimal("0")


def test_document_non_facture_produit_le_payload_d_echec(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un document parfaitement lisible mais qui n'est pas une facture.

    L'extraction de texte réussit, le modèle répond correctement — il n'a
    simplement rien trouvé. Sans aucun total ni ligne, il n'y a rien qu'un
    humain puisse corriger : c'est un échec, pas une extraction pauvre.
    """
    modele.reponse = llm_fixtures.reponse_vide_de_sens()

    assert _poster(documents.document_non_facture_pdf()) == 202

    assert callbacks[0].score_confiance == Decimal("0")


# --- Validation du modèle : réponses fautives ------------------------------


def test_appel_au_modele_en_echec_produit_le_payload_d_echec(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Groq injoignable (timeout, quota, clé invalide) → verdict d'échec émis.

    Le document ne doit jamais rester bloqué « en attente » côté API data parce
    que le fournisseur de modèle était indisponible.
    """
    modele.erreur = LlmClientError("appel Groq échoué")

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].score_confiance == Decimal("0")


@pytest.mark.parametrize(
    ("cas", "reponse"),
    [
        ("json_tronque", llm_fixtures.JSON_TRONQUE),
        ("reponse_vide", llm_fixtures.REPONSE_VIDE),
        ("json_non_objet", llm_fixtures.JSON_NON_OBJET),
    ],
)
def test_reponse_modele_inexploitable_produit_le_payload_d_echec(
    modele: _ModeleSimule,
    callbacks: list[OcrWebhookPayload],
    cas: str,
    reponse: str,
) -> None:
    """Une sortie de modèle non parsable ne fait pas planter la tâche de fond.

    Trois façons dont un modèle dérive malgré un schéma strict : réponse coupée
    par la limite de tokens, réponse vide, ou JSON valide mais qui n'est pas un
    objet. Toutes doivent aboutir au même verdict exploitable côté API data.
    """
    modele.reponse = reponse

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].score_confiance == Decimal("0")


def test_champ_inattendu_du_modele_n_atteint_pas_le_contrat(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un champ inventé par le modèle est ignoré, sans faire échouer l'extraction.

    Le contrat partagé avec l'API data est une liste fermée : ce que le modèle
    ajoute en trop ne doit ni s'y glisser, ni casser le reste de l'extraction.
    """
    modele.reponse = llm_fixtures.reponse_facture(
        tva_intracommunautaire="FR00111222333",
        mention_speciale="autoliquidation",
    )

    assert _poster(documents.facture_native_pdf()) == 202

    payload = callbacks[0]
    assert payload.score_confiance > 0  # l'extraction aboutit normalement
    assert payload.numero_facture == documents.NUMERO_FACTURE
    # Le champ surnuméraire n'existe pas dans le payload transmis.
    assert "tva_intracommunautaire" not in payload.model_dump()
    assert "mention_speciale" not in payload.model_dump_json()


def test_champ_contrat_manquant_reste_exploitable(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un champ du contrat absent de la réponse est traité comme non extrait.

    Le schéma strict l'impose, mais un modèle peut l'omettre : l'extraction doit
    se poursuivre, le champ vaut ``None`` au contrat et 0 dans le détail par
    champ (l'humain saura qu'il n'a rien été lu).
    """
    donnees = json.loads(llm_fixtures.reponse_facture())
    del donnees["numero_facture"]
    modele.reponse = json.dumps(donnees)

    assert _poster(documents.facture_native_pdf()) == 202

    payload = callbacks[0]
    assert payload.score_confiance > 0
    assert payload.numero_facture is None
    assert payload.par_champ is not None
    assert payload.par_champ["numero_facture"] == Decimal("0")


def test_valeur_de_type_incoherent_produit_le_payload_d_echec(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Un montant renvoyé en toutes lettres ne construit pas le contrat.

    Le JSON est valide, le champ existe, mais sa valeur ne se convertit pas en
    montant : la validation contrat échoue et le pipeline émet le verdict
    d'échec plutôt qu'un payload à moitié faux.
    """
    modele.reponse = llm_fixtures.reponse_facture(total_ht="mille euros")

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].score_confiance == Decimal("0")


def test_date_echeance_derivee_du_delai_de_paiement(
    modele: _ModeleSimule, callbacks: list[OcrWebhookPayload]
) -> None:
    """Une échéance absente mais un délai lu (« net 30 ») est dérivée en Python.

    L'arithmétique n'est pas confiée au modèle : on vérifie ici qu'elle est bien
    faite dans le pipeline et que le résultat atteint le contrat.
    """
    modele.reponse = llm_fixtures.reponse_facture(
        date_echeance=None,
        delai_paiement_jours=30,
    )

    assert _poster(documents.facture_native_pdf()) == 202

    assert callbacks[0].date_echeance == documents.DATE_EMISSION + timedelta(days=30)
