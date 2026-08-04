# ==============================================================================
# Image locale de l'API IA Factur-IA (FastAPI / Python 3.13).
# Une seule étape : aucune compilation, toutes les dépendances lourdes (torch,
# PyMuPDF, opencv-headless) sont livrées en roues autonomes — pas de paquet apt.
# ==============================================================================

FROM python:3.13-slim

# uv (gestionnaire de paquets) copié depuis son image officielle.
# Version épinglée pour des builds reproductibles, sans `curl | sh`.
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# PYTHONUNBUFFERED : les logs sortent immédiatement (visibles dans
# `docker compose logs`). UV_PROJECT_ENVIRONMENT : le venv est créé dans
# /opt/venv, HORS de /app, car en dev le projet est monté dans /app depuis
# l'hôte — un venv interne au projet serait masqué par ce montage.
ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dépendances installées avant la copie du code : tant que pyproject.toml et
# uv.lock ne changent pas, le cache Docker réutilise cette couche.
# --no-install-project : seules les dépendances sont installées, le code n'est
# pas encore là. Il n'a de toute façon pas à l'être : uvicorn importe `src`
# depuis /app, monté depuis l'hôte.
# torch est résolu depuis l'index CPU (cf. pyproject.toml) : sans cela, les
# roues CUDA ajouteraient ~4 Go à l'image pour un GPU jamais utilisé.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Code de l'application (filtré par .dockerignore : ni .env, ni .venv, ni .git).
COPY . .

# Les poids EasyOCR (~98 Mo) sont téléchargés au premier document scanné, dans
# ce répertoire monté sur un volume par le compose : le téléchargement n'a lieu
# qu'une fois et survit aux reconstructions. Tant qu'il est vide, `GET /ready`
# répond 503 — c'est le rôle de la sonde.
# En production (système de fichiers éphémère, instances froides), les cuire
# plutôt dans l'image ici même, par un `RUN` qui instancie easyocr.Reader.
ENV EASYOCR_MODULE_PATH=/models

EXPOSE 8001

# --no-sync : l'environnement est déjà construit ci-dessus, rien à
# resynchroniser au démarrage du conteneur.
CMD ["uv", "run", "--no-sync", "uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]
