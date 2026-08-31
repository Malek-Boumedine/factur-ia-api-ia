"""Moteur de recherche sémantique sur la FAQ réglementaire (POC).

Principe : les questions du corpus sont encodées en vecteurs par un modèle
d'embeddings multilingue local, la requête de l'utilisateur aussi, et la
recherche est une similarité cosinus exacte en mémoire — un produit scalaire
numpy sur des vecteurs normalisés. Pas de FAISS : à cette échelle (et jusqu'à
~100 000 entrées), un index approché n'apporterait rien qu'un calcul exact ne
fasse déjà en millisecondes.

On vectorise la **question seule** : la requête de l'utilisateur est une
question, l'appariement question ↔ question est homogène (même longueur, même
registre). Limite assumée : une requête portant sur un détail présent
uniquement dans la réponse peut passer à côté — cf. docs/poc-recherche-semantique.md.

Découpage pensé pour les tests : la partie pure (lecture du CSV, classement
par similarité) est testable sans modèle ni réseau ; seul ``SearchEngine``
charge le modèle (téléchargé au premier usage dans le cache Hugging Face).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Couplage volontairement limité au fichier : mêmes chemin et colonnes que le
# CSV produit par src/scraping/storage.py, sans importer le code du scraping.
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "faq.csv"

MODEL_NAME = "intfloat/multilingual-e5-small"
# Les modèles e5 attendent un préfixe indiquant le rôle du texte. Pour un
# appariement symétrique question ↔ question : « query: » des deux côtés.
E5_PREFIX = "query: "


@dataclass(frozen=True)
class FaqDocument:
    """Une entrée du corpus, réduite aux champs utiles à la restitution."""

    question: str
    reponse: str
    source: str
    url: str


@dataclass(frozen=True)
class SearchResult:
    """Un document du corpus et sa similarité cosinus avec la requête (0 à 1)."""

    document: FaqDocument
    score: float


def load_documents(csv_path: Path = CSV_PATH) -> list[FaqDocument]:
    """Charge les questions-réponses du CSV produit par le module de scraping."""
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        return [
            FaqDocument(
                question=row["question"],
                reponse=row["reponse"],
                source=row["source"],
                url=row["url"],
            )
            for row in csv.DictReader(csv_file)
        ]


def rank_by_similarity(
    query_vector: npt.NDArray[np.float32],
    document_vectors: npt.NDArray[np.float32],
    top_k: int,
) -> list[tuple[int, float]]:
    """Classe les documents par similarité cosinus décroissante avec la requête.

    Les vecteurs sont supposés normalisés (norme 1) : la similarité cosinus se
    réduit alors au produit scalaire. Renvoie les ``top_k`` meilleurs sous
    forme de couples (indice du document, score).
    """
    scores = document_vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in best_indices]


class SearchEngine:
    """Corpus encodé en mémoire, prêt à répondre à des requêtes.

    Les embeddings sont recalculés à chaque instanciation : à 13 entrées,
    l'encodage prend moins d'une seconde — c'est le chargement du modèle qui
    domine, et une persistance sur disque ne l'éviterait pas.
    """

    def __init__(self, csv_path: Path = CSV_PATH) -> None:
        # Import local : sentence-transformers (et torch derrière) est lourd à
        # importer, et le groupe `poc` peut être absent — le reste du module
        # (partie pure) doit rester importable sans lui.
        from sentence_transformers import SentenceTransformer

        self.documents = load_documents(csv_path)
        self._model = SentenceTransformer(MODEL_NAME)
        self._document_vectors = self._encode(
            [document.question for document in self.documents]
        )

    def _encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode des textes en vecteurs normalisés (préfixe e5 inclus)."""
        vectors = self._model.encode(
            [E5_PREFIX + text for text in texts], normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Renvoie les ``top_k`` entrées du corpus les plus proches de la requête."""
        query_vector = self._encode([query])[0]
        ranked = rank_by_similarity(query_vector, self._document_vectors, top_k)
        return [
            SearchResult(document=self.documents[index], score=score)
            for index, score in ranked
        ]
