"""Configuration pytest à la racine des tests.

Charge le fichier `.env.test` (valeurs factices) AVANT tout import de
`src.core.config`, car `Settings()` est instancié à l'import du module de
config : sans ces variables, l'instanciation échoue et la collecte pytest
plante en CI (aucune variable d'environnement fournie par le runner).
"""

from pathlib import Path

from dotenv import load_dotenv

# Chargé au niveau module (avant tout import de src.core.config par les tests).
_ENV_TEST = Path(__file__).parent.parent / ".env.test"
load_dotenv(_ENV_TEST, override=True)