"""Configuration de l'application via Pydantic Settings.

Les variables sont lues depuis l'environnement (ou le fichier .env en local).
`Settings()` est instancié à l'import : toute variable requise manquante *ou
vide* fait échouer le démarrage — c'est voulu (fail-fast sur une config
incomplète).
"""

from decimal import Decimal
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Variable indispensable au fonctionnement : absente *ou vide*, le démarrage
# échoue. Sans `min_length`, une variable définie à `""` passe la validation et
# la panne n'apparaît qu'au premier appel réel (extraction, callback).
Requis = Annotated[str, Field(min_length=1)]


class Settings(BaseSettings):
    """Variables d'environnement de l'API IA."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "factur-ia-api-ia"
    ENVIRONNEMENT: str = "dev"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"  # noqa: S104  # nosec B104
    API_PORT: int = 8090

    # --- Sécurité (token partagé avec l'API data pour le callback OCR) ---
    SECRET_OCR_TOKEN: Requis

    # --- API data (callback) ---
    DATA_API_BASE_URL: Requis
    HTTP_TIMEOUT_SECONDS: float = 30.0
    HTTP_MAX_RETRIES: int = 3
    # Authentification IAM Cloud Run du callback (jeton d'identité Google dans
    # X-Serverless-Authorization, cf. core/gcp_identity.py). Symétrique de
    # IA_API_IAM_AUTH_ENABLED côté API data : chaque service nomme le service
    # appelé. Défaut faux : en dev/test, l'API data locale n'exige aucun jeton
    # et il n'y a pas de serveur de métadonnées.
    DATA_API_IAM_AUTH_ENABLED: bool = False

    # --- LLM Groq ---
    GROQ_API_KEY: Requis
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TIMEOUT_SECONDS: float = 60.0

    # --- OCR ---
    OCR_LANGUAGES: str = "fr,en"
    EASYOCR_GPU: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10

    # --- CORS ---
    CORS_ORIGINS: str = "*"

    # --- Observabilité (OpenTelemetry) ---
    # Deux interrupteurs indépendants, désactivés par défaut : rien n'est
    # instrumenté en local ni en CI. OTEL_ENABLED pilote les traces (export
    # OTLP), OTEL_METRICS_ENABLED les métriques — dont celles de qualité
    # d'extraction — exposées sur /metrics au format Prometheus (jamais public
    # en production). (Noms maison, alignés sur l'API data : la variable
    # standard OTEL_SDK_DISABLED a une sémantique inversée avec défaut =
    # activé.) Les autres variables suivent les conventions standard
    # OpenTelemetry ; le SDK les lit dans os.environ, pas dans le .env —
    # setup_telemetry les y relaie, l'environnement réel restant prioritaire.
    OTEL_ENABLED: bool = False
    OTEL_METRICS_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "factur-ia-api-ia"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_TRACES_EXPORTER: str = "otlp"
    OTEL_TRACES_SAMPLER: str | None = None
    OTEL_TRACES_SAMPLER_ARG: str | None = None
    # Conventions sémantiques HTTP stables : requises pour le label
    # http.route des métriques (découpage par route).
    OTEL_SEMCONV_STABILITY_OPT_IN: str = "http"

    # --- Monitoring de la qualité d'extraction (MLflow) ---
    # Désactivé par défaut, comme les interrupteurs d'observabilité de l'API data :
    # sans activation explicite, rien n'est tracé et ``mlflow`` n'est même pas
    # importé (local et CI strictement inchangés). L'URI par défaut est un simple
    # fichier SQLite local : aucun serveur n'est requis pour écrire, l'interface
    # (``mlflow ui``) ne sert qu'à relire. Le store « répertoire de fichiers »
    # (``file:./mlruns``) existe encore mais MLflow l'a placé en maintenance —
    # SQLite est le backend local recommandé.
    MLFLOW_ENABLED: bool = False
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "factur-ia-extraction"
    # Score de confiance sous lequel une extraction est signalée comme dégradée
    # (WARNING applicatif + tag ``alerte`` sur le run). Au-dessus de 0.6, borne
    # haute imposée par le malus d'incohérence de ``confidence.py``.
    MONITORING_SEUIL_ALERTE: Decimal = Decimal("0.7")

    @property
    def ocr_callback_url(self) -> str:
        """URL complète du webhook OCR de l'API data."""
        return f"{self.DATA_API_BASE_URL.rstrip('/')}/documents/webhook/ocr"

    @property
    def ocr_languages_list(self) -> list[str]:
        """Langues OCR sous forme de liste (pour EasyOCR)."""
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(",") if lang.strip()]


settings = Settings()
