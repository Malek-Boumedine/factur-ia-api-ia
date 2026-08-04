"""Point d'entrée de l'API IA d'extraction de factures."""

import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.extractions.ocr_extractor import ocr_model_available
from src.extractions.router import router as extractions_router

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure le logging applicatif (uvicorn ne configure que SES loggers).

    Sans cette configuration, le logger racine n'a aucun handler : les logs
    ``INFO`` du pipeline (extraction démarrée/réussie, callback accepté) sont
    perdus, et les ``WARNING``/``ERROR`` sortent bruts (handler de dernier
    recours de Python), sans horodatage ni nom de logger — diagnostic impossible
    en production. ``basicConfig`` est sans effet si le logger racine a déjà des
    handlers (configuration posée par le déployeur) : on n'écrase rien.

    Niveau : ``INFO`` par défaut, ``DEBUG`` si ``settings.DEBUG`` est actif —
    active les logs de diagnostic du pipeline (ex. JSON structuré avant
    validation). Dev uniquement.
    """
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


_configure_logging()

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(extractions_router)


# --- Sondes de disponibilité ------------------------------------------------
#
# Routes d'infrastructure, destinées à la plateforme de déploiement : publiques
# (Cloud Run sonde sans en-tête d'authentification), mais hors contrat OpenAPI
# (``include_in_schema=False``) — personne n'a à les consommer comme des routes
# métier — et sans aucune information exploitable dans les réponses (ni version,
# ni configuration, ni détail d'erreur).
#
# ATTENTION si une instrumentation HTTP (OpenTelemetry, Prometheus) est ajoutée
# ici plus tard : ces deux routes doivent en être EXCLUES. Sondées en continu,
# elles représenteraient l'essentiel du volume de requêtes et écraseraient les
# statistiques de latence et de taux d'erreur du trafic réel. Le monitoring
# actuel (MLflow) ne trace que la qualité d'extraction, document par document :
# les sondes ne le touchent jamais, il n'y a rien à exclure aujourd'hui.


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Sonde de liveness : le processus est-il vivant ?

    Répond 200 inconditionnellement — aucune I/O, aucune dépendance, pas même
    une lecture de disque. Un échec de cette sonde provoque le **redémarrage**
    du conteneur : elle ne doit donc jamais dépendre de quoi que ce soit
    d'externe, sous peine de faire redémarrer en boucle des instances saines
    parce qu'un tiers est tombé.
    """
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def ready() -> dict[str, str]:
    """Sonde de readiness : cette instance peut-elle mener une extraction à bien ?

    Un échec ici **retire l'instance du trafic sans la tuer** ; elle y revient
    d'elle-même dès que la sonde repasse au vert. La règle appliquée : ne sortir
    du trafic que si la panne est **locale à l'instance** et qu'une autre
    instance ferait mieux.

    Une seule dépendance satisfait ce critère — les poids EasyOCR, vérifiés sur
    disque sans réseau ni chargement de torch. Ne sont volontairement **pas**
    vérifiés :

    - **Groq** : aucun appel depuis une sonde interrogée en continu (service
      payant, quota). Surtout, se retirer du trafic parce que Groq est tombé
      nous priverait d'émettre les payloads d'échec, et les documents
      resteraient bloqués « en attente » côté API data au lieu de passer
      proprement en « erreur ». La présence de la clé n'est pas testée non plus :
      ``GROQ_API_KEY`` est requise par ``Settings``, donc l'application ne
      démarre pas sans elle.
    - **l'API data** (destination du callback) : panne partagée, non locale. Le
      callback a ses propres retries, il intervient en fin de pipeline et pas à
      l'entrée, et si l'API data est indisponible elle ne nous envoie plus rien
      — il n'y a aucun trafic à retirer.

    Le motif du refus part dans les logs, jamais dans la réponse.
    """
    if not ocr_model_available():
        logger.warning(
            "Sonde de readiness : aucun poids OCR installé localement, "
            "instance non prête."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service non prêt",
        )
    return {"status": "ready"}
