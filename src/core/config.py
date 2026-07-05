"""Configuration de l'application via Pydantic Settings.

Les variables sont lues depuis l'environnement (ou le fichier .env en local).
`Settings()` est instancié à l'import : toute variable requise manquante fait
échouer le démarrage — c'est voulu (fail-fast sur une config incomplète).
"""

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
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TIMEOUT_SECONDS: float = 60.0

    # --- OCR ---
    OCR_LANGUAGES: str = "fr,en"
    EASYOCR_GPU: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10

    # --- CORS ---
    CORS_ORIGINS: str = "*"

    @property
    def ocr_callback_url(self) -> str:
        """URL complète du webhook OCR de l'API data."""
        return f"{self.DATA_API_BASE_URL.rstrip('/')}/documents/webhook/ocr"

    @property
    def ocr_languages_list(self) -> list[str]:
        """Langues OCR sous forme de liste (pour EasyOCR)."""
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(",") if lang.strip()]


settings = Settings()
