"""Interface Streamlit minimale du POC de recherche sémantique.

Usage : ``uv run streamlit run src/recherche/app.py``.
Un champ de saisie, les résultats avec score, source et réponse — rien de plus.
Le moteur (modèle + corpus encodé) est gardé en ``session_state`` pour n'être
chargé qu'une fois par session, pas à chaque requête.
"""

import streamlit as st

from src.recherche.engine import SearchEngine


def get_engine() -> SearchEngine:
    """Charge le moteur une seule fois par session Streamlit.

    Cache manuel via ``session_state`` plutôt que ``@st.cache_resource`` : le
    décorateur, non typé dans l'environnement du hook mypy de pre-commit,
    ferait échouer le mode strict.
    """
    if "engine" not in st.session_state:
        st.session_state["engine"] = SearchEngine()
    engine: SearchEngine = st.session_state["engine"]
    return engine


st.set_page_config(page_title="POC recherche sémantique — Factur-IA")
st.title("Recherche sémantique — FAQ facturation électronique")
st.caption(
    "POC : interrogation en langage naturel du corpus collecté par le module "
    "de scraping (data/faq.csv). Modèle d'embeddings local, aucun appel externe."
)

query = st.text_input("Votre question", placeholder="Ex : quand est-ce obligatoire ?")

if query:
    with st.spinner("Recherche…"):
        results = get_engine().search(query)
    for result in results:
        st.subheader(result.document.question)
        st.markdown(
            f"**Score : {result.score:.3f}** — "
            f"[{result.document.source}]({result.document.url})"
        )
        st.write(result.document.reponse)
        st.divider()
