"""Client LLM Groq : appel du modèle pour structurer le texte extrait.

Dernière brique du pipeline avant le callback : elle reçoit le texte brut issu
de l'extraction (PDF natif ou OCR) et le soumet à un modèle Groq pour obtenir une
réponse structurée. Ce module ne fait *que* l'appel au modèle : il expose une
fonction générique ``call_llm(system_prompt, user_content) -> texte brut``.

Ce module ne connaît ni le prompt métier de structuration, ni le parsing du JSON
renvoyé (étapes ultérieures). Il ne fige pas non plus le format de réponse
(``response_format``) : le mode structured outputs / json_schema sera branché à
la tâche de structuration. ``openai/gpt-oss-120b`` (défaut ``GROQ_MODEL``)
supporte nativement les structured outputs stricts côté Groq.

Comme le ``Reader`` EasyOCR, le client Groq (coûteux à instancier) est construit
une seule fois puis mis en cache au niveau module. Le SDK ``groq`` est importé
paresseusement, pour ne pas payer l'import tant qu'aucun appel n'est demandé.

La clé d'API est lue *uniquement* depuis ``settings.GROQ_API_KEY`` : jamais en
dur, jamais journalisée (ni la clé, ni le contenu des requêtes).
"""

from typing import Any

from src.core.config import settings

# Extraction déterministe (pas créative) : température au plus bas pour une sortie
# stable et reproductible. Détail d'implémentation (pas dans Settings).
_TEMPERATURE = 0.0

# Client Groq mis en cache : construit une seule fois (voir ``_get_client``).
_client: Any = None


class LlmClientError(Exception):
    """Appel au LLM Groq impossible ou échoué.

    Relevée pour toute défaillance de l'appel : clé invalide, timeout, erreur
    réseau, rate limit, ou toute erreur renvoyée par l'API Groq. L'orchestrateur
    du pipeline attrape cette exception pour produire un résultat d'échec
    (``score_confiance = 0``) côté API data.
    """


def _get_client() -> Any:  # le SDK groq n'expose pas de stubs typés
    """Renvoie le client Groq, construit une seule fois puis mis en cache.

    L'instanciation (coûteuse) n'a lieu qu'au premier appel, avec la clé
    (``GROQ_API_KEY``) et le timeout (``GROQ_TIMEOUT_SECONDS``) de la
    configuration. Le SDK ``groq`` est importé ici (et non en tête de module)
    pour ne pas payer l'import tant qu'aucun appel LLM n'est réellement demandé.
    """
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(
            api_key=settings.GROQ_API_KEY,
            timeout=settings.GROQ_TIMEOUT_SECONDS,
        )
    return _client


def call_llm(system_prompt: str, user_content: str) -> str:
    """Soumet un prompt système + un contenu utilisateur au modèle Groq.

    Appel non-streamé (``stream=False`` : réponse complète d'un coup) et
    déterministe (``temperature=0``, adapté à l'extraction). Le modèle
    (``GROQ_MODEL``) et le timeout viennent de la configuration. Aucun format de
    réponse n'est imposé : la fonction renvoie le texte brut du modèle, à charger
    de structurer/parser en aval.

    Args:
        system_prompt: instruction système (rôle, consignes) donnée au modèle.
        user_content: contenu utilisateur soumis (ici, le texte brut extrait).

    Returns:
        Le contenu textuel brut de la réponse du modèle (chaîne vide si le modèle
        ne renvoie aucun contenu).

    Raises:
        LlmClientError: échec de l'appel (clé invalide, timeout, réseau, rate
            limit, erreur API) → extraction inexploitable, ``score_confiance = 0``
            côté API data.
    """
    try:
        response = _get_client().chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=_TEMPERATURE,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # SDK Groq : APIError, timeout, rate limit, réseau...
        raise LlmClientError(
            "Appel au LLM Groq échoué : structuration impossible."
        ) from exc
