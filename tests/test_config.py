"""Tests de la configuration (`src/core/config.py`).

On vérifie le fail-fast : une variable indispensable définie mais **vide** doit
faire échouer l'instanciation des settings, donc le démarrage de l'application.
Sans cette garantie, la panne serait différée au premier appel réel (clé Groq
vide → erreur d'authentification à la première extraction seulement).
"""

import pytest
from pydantic import ValidationError
from src.core.config import Settings

# Ces trois variables n'ont pas de valeur par défaut : sans elles, l'API ne peut
# ni appeler le LLM, ni authentifier l'API data, ni lui renvoyer le résultat.
VARIABLES_REQUISES = ["GROQ_API_KEY", "SECRET_OCR_TOKEN", "DATA_API_BASE_URL"]


@pytest.mark.parametrize("variable", VARIABLES_REQUISES)
def test_variable_requise_vide_refusee(variable: str) -> None:
    """Une chaîne vide est rejetée au même titre qu'une variable absente."""
    # Passées à l'instanciation, ces valeurs priment sur l'environnement et le
    # fichier .env : le test ne dépend pas de la config du poste.
    with pytest.raises(ValidationError) as erreur:
        Settings(**{variable: ""})  # type: ignore[arg-type]

    assert variable in str(erreur.value)


def test_variables_requises_renseignees_acceptees() -> None:
    """Contre-épreuve : des valeurs non vides passent la validation."""
    # Valeurs factices de test : le S106 de ruff est un faux positif ici, comme
    # le `# pragma` l'est pour detect-secrets.
    settings = Settings(
        GROQ_API_KEY="cle-groq",  # pragma: allowlist secret
        SECRET_OCR_TOKEN="token-ocr",  # noqa: S106  # pragma: allowlist secret
        DATA_API_BASE_URL="http://localhost:8000",
    )

    assert settings.GROQ_API_KEY == "cle-groq"  # pragma: allowlist secret
