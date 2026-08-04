"""Configuration de l'application via Pydantic Settings.

Les variables sont lues depuis l'environnement (ou le fichier .env en local).
`Settings()` est instancié à l'import : toute variable requise manquante fait
échouer le démarrage — c'est voulu (fail-fast sur une config incomplète).
"""

from decimal import Decimal

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    SECRET_OCR_TOKEN: str

    # --- API data (callback) ---
    DATA_API_BASE_URL: str
    HTTP_TIMEOUT_SECONDS: float = 30.0
    HTTP_MAX_RETRIES: int = 3

    # --- LLM Groq ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TIMEOUT_SECONDS: float = 60.0

    # --- OCR ---
    OCR_LANGUAGES: str = "fr,en"
    EASYOCR_GPU: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10

    # --- CORS ---
    CORS_ORIGINS: str = "*"

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
