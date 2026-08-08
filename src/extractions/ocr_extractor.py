"""Extraction du texte brut par OCR (EasyOCR) : images et PDF scannés.

Branche « OCR » du pipeline, en parallèle de l'extraction PDF natif. Elle traite
les documents sans couche texte extractible :

- images directes (JPEG/PNG), routées ici par l'orchestrateur ;
- PDF classés ``SCANNED`` par ``detect_pdf_type`` (images encapsulées dans un
  PDF) : EasyOCR ne lit que des images, on convertit donc d'abord chaque page en
  image (PyMuPDF) avant l'OCR.

Ce module ne fait *que* l'OCR : ni détection, ni structuration LLM. Le texte brut
reconnu est concaténé dans l'ordre des pages pour être structuré ensuite par le
LLM (étape ultérieure).

Deux optimisations volontaires :

- ``easyocr`` est importé paresseusement (dans ``_get_reader``) : importer ce
  module ne charge donc pas torch.
- le ``Reader`` EasyOCR (chargement des modèles, coûteux) est construit une seule
  fois et mis en cache au niveau module, puis réutilisé à chaque appel.

Ce module expose aussi ``ocr_model_available()``, qui répond « les poids OCR
sont-ils installés ? » sans rien charger : c'est le seul contrôle de la sonde de
readiness (``GET /ready``). Il vit ici parce que c'est ce module qui connaît
EasyOCR — ``main.py`` n'a pas à savoir où sont rangés les poids.
"""

import os
from pathlib import Path
from typing import Any

import fitz

from src.core.config import settings

# Résolution de rendu des pages de PDF scanné en image, avant OCR. 300 DPI est un
# bon compromis qualité de reconnaissance / taille. Détail d'implémentation (pas
# dans Settings) ; pourra être remonté en configuration si un ajustement sans
# redéploiement devient nécessaire.
_RENDER_DPI = 300

# Plafond de pages pour l'OCR d'un PDF scanné. Borne à la fois le temps de calcul
# (``readtext`` est CPU-bound, sans timeout possible dans un thread — un vrai
# timeout exigerait un process pool, cf. dette BackgroundTasks) et la mémoire
# (chaque page rendue à 300 DPI pèse ~25 Mo en PNG, toutes matérialisées d'un
# coup). Largement au-dessus d'une facture réelle ; au-delà, le document est
# déclaré inexploitable (payload d'échec côté API data). Détail d'implémentation
# (pas dans Settings) ; pourra être remonté en configuration si besoin.
_MAX_OCR_PAGES = 20

# Reader EasyOCR mis en cache : construit une seule fois (voir ``_get_reader``).
_reader: Any = None


class OcrExtractionError(Exception):
    """Image ou PDF scanné illisible, ou aucun texte reconnu par l'OCR.

    Relevée quand PyMuPDF ne parvient pas à ouvrir le PDF, quand EasyOCR échoue
    sur une image corrompue, quand le PDF dépasse le plafond de pages
    (``_MAX_OCR_PAGES``), ou quand l'OCR ne reconnaît finalement aucun texte.
    L'orchestrateur du pipeline attrape cette exception pour produire un résultat
    d'échec (``score_confiance = 0``) côté API data.
    """


def _get_reader() -> Any:
    """Renvoie le ``Reader`` EasyOCR, construit une seule fois puis mis en cache.

    Le chargement des modèles est coûteux : on ne l'initialise qu'au premier
    appel, avec les langues (``OCR_LANGUAGES``) et le mode GPU (``EASYOCR_GPU``)
    de la configuration. ``easyocr`` est importé ici (et non en tête de module)
    pour éviter de charger torch tant qu'aucun OCR n'est réellement demandé.
    """
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(settings.ocr_languages_list, gpu=settings.EASYOCR_GPU)
    return _reader


def _model_storage_directory() -> Path:
    """Renvoie le répertoire où EasyOCR range ses poids, résolu comme elle le fait.

    Reflet volontaire d'un détail interne d'``easyocr`` (``easyocr/config.py``) :
    on ne peut pas le lire en important la bibliothèque, puisque cet import charge
    torch — inacceptable depuis une sonde interrogée en continu. Les deux
    variables d'environnement consultées, et leur ordre de priorité, sont ceux
    d'``easyocr`` ; à resynchroniser si elle change sa résolution.
    """
    base = (
        os.environ.get("EASYOCR_MODULE_PATH")
        or os.environ.get("MODULE_PATH")
        or os.path.expanduser("~/.EasyOCR/")
    )
    return Path(base) / "model"


def ocr_model_available() -> bool:
    """Indique si des poids OCR sont installés localement, sans rien charger.

    Support de la sonde de readiness (``GET /ready``). Sans ces poids, le premier
    document scanné déclencherait leur téléchargement (~98 Mo) *au milieu* du
    pipeline : sur un système de fichiers éphémère, cette attente non bornée se
    rejoue à chaque instance froide, et une indisponibilité du CDN se traduirait
    en extraction ratée (``score_confiance = 0``) pour un document pourtant
    lisible. C'est la seule dépendance du service qui soit locale à l'instance —
    donc la seule dont la panne justifie de sortir du trafic.

    On teste la **présence d'au moins un fichier de poids**, pas des noms
    précis : le mode de panne réel est un répertoire vide (image construite sans
    les poids, instance fraîche), et ce critère ne casse ni quand ``easyocr``
    renomme ses fichiers, ni quand ``OCR_LANGUAGES`` change de réseau de
    reconnaissance.

    Une sonde ne doit jamais lever, et celle-ci ne le peut pas : ``Path.glob``
    absorbe les erreurs du système de fichiers (répertoire absent, droits
    refusés, parent qui n'est pas un répertoire) et renvoie simplement un
    résultat vide — tous ces cas se traduisent donc en « indisponible », sans
    garde explicite à écrire.

    Returns:
        ``True`` si au moins un fichier de poids est présent, ``False`` sinon
        (répertoire vide, absent, ou illisible).
    """
    return any(_model_storage_directory().glob("*.pth"))


def _ocr_image(image: bytes) -> str:
    """Reconnaît le texte d'une image (bytes encodés) via EasyOCR.

    ``readtext(detail=0)`` renvoie la liste des fragments texte dans l'ordre de
    lecture, joints par un saut de ligne. EasyOCR accepte directement des bytes
    d'image encodée : pas de conversion numpy/PIL à gérer.
    """
    try:
        fragments: list[str] = _get_reader().readtext(image, detail=0)
    except Exception as exc:  # EasyOCR lève des exceptions variées et peu typées
        raise OcrExtractionError(
            "Image illisible ou corrompue : OCR impossible."
        ) from exc
    return "\n".join(fragment for fragment in fragments if fragment.strip())


def _pdf_to_images(content: bytes) -> list[bytes]:
    """Convertit chaque page d'un PDF scanné en image PNG (bytes), dans l'ordre.

    EasyOCR ne lit que des images : on rend chaque page à ``_RENDER_DPI`` avec
    PyMuPDF, autonome (aucune dépendance système type poppler). Un PDF dépassant
    ``_MAX_OCR_PAGES`` est rejeté avant tout rendu (borne temps + mémoire).
    """
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            if doc.page_count > _MAX_OCR_PAGES:
                raise OcrExtractionError(
                    f"PDF scanné de {doc.page_count} pages : plafond OCR de "
                    f"{_MAX_OCR_PAGES} pages dépassé, document inexploitable."
                )
            return [page.get_pixmap(dpi=_RENDER_DPI).tobytes("png") for page in doc]
    except OcrExtractionError:
        raise
    except Exception as exc:  # PyMuPDF lève des exceptions variées et peu typées
        raise OcrExtractionError(
            "PDF scanné illisible ou corrompu : conversion en image impossible."
        ) from exc


def extract_ocr_text(content: bytes, *, is_pdf: bool) -> str:
    """Extrait le texte brut d'une image ou d'un PDF scanné par OCR.

    Pour un PDF scanné, chaque page est d'abord convertie en image, puis passée à
    l'OCR ; les textes sont concaténés dans l'ordre des pages (séparés par une
    ligne vide, comme pour l'extraction PDF natif). Pour une image, l'OCR est
    appliqué directement.

    Args:
        content: contenu binaire du fichier (image JPEG/PNG, ou PDF scanné).
        is_pdf: ``True`` si ``content`` est un PDF scanné (conversion en images
            requise), ``False`` s'il s'agit d'une image directe.

    Returns:
        Le texte brut reconnu, concaténé dans l'ordre des pages.

    Raises:
        OcrExtractionError: PDF/image illisible, PDF dépassant ``_MAX_OCR_PAGES``,
            ou aucun texte reconnu (document inexploitable → ``score_confiance = 0``
            côté API data).
    """
    if is_pdf:
        pages_text = [_ocr_image(image) for image in _pdf_to_images(content)]
        text = "\n\n".join(page for page in pages_text if page).strip()
    else:
        text = _ocr_image(content).strip()

    if not text:
        raise OcrExtractionError(
            "Aucun texte reconnu par l'OCR : extraction inexploitable."
        )

    return text
