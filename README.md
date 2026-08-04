# factur-ia-api-ia

Microservice IA d'extraction de factures du système **Factur-IA**. Il reçoit un
PDF ou une image de l'API data, en extrait le texte (pdfplumber pour un PDF
natif, EasyOCR pour un scan ou une image), le structure via un LLM (Groq), puis
renvoie le résultat à l'API data par un callback signé.

## Démarrage

```bash
uv sync --all-groups                          # dépendances
uv run uvicorn src.main:app --reload --port 8001
```

Copier `.env.example` en `.env` et renseigner au minimum `SECRET_OCR_TOKEN`
(partagé avec l'API data) et `GROQ_API_KEY` : sans elles, l'application ne
démarre pas.

```bash
uv run pytest --cov=src                       # tests
uv run mypy src/                              # typage strict
uv run pre-commit run --all-files             # lint + format
```

## Tests

210 tests, 100 % de couverture de `src/`. La suite tourne en une dizaine de
secondes, **sans réseau** : le LLM Groq, EasyOCR et le callback de l'API data
sont toujours simulés, et une garde installée dans `tests/conftest.py` fait
échouer tout test qui tenterait une connexion réelle. Les documents d'exemple
sont générés en mémoire (aucun binaire versionné, aucune donnée réelle : les
SIRET et IBAN sont inventés).

La **[stratégie de test](docs/strategie-de-test.md)** détaille, pour chaque étape
du pipeline, la partie visée, le périmètre, l'approche retenue et les limites
connues — notamment l'absence de vérité terrain, qui interdit toute mesure du
taux d'erreur d'extraction.

## Monitoring de la qualité d'extraction

### À quoi ça sert

Chaque extraction produit déjà des signaux de qualité : un score de confiance
global, une confiance par champ, un type de document suggéré. Le monitoring les
**trace dans le temps** pour répondre à trois questions :

- la qualité du modèle se dégrade-t-elle (dérive) ?
- quels champs sont chroniquement mal extraits (→ retoucher le prompt) ?
- un changement de modèle Groq améliore-t-il ou dégrade-t-il les résultats ?

C'est du monitoring **de modèle**, distinct du monitoring **applicatif**
(latence, erreurs HTTP) assuré par la stack OpenTelemetry de l'API data.

### Activation

Désactivé par défaut. Quatre variables, toutes documentées dans `.env.example` :

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `MLFLOW_ENABLED` | `False` | Interrupteur unique du traçage |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Où sont stockées les métriques |
| `MLFLOW_EXPERIMENT_NAME` | `factur-ia-extraction` | Nom de l'expérience MLflow |
| `MONITORING_SEUIL_ALERTE` | `0.7` | Score sous lequel une extraction est signalée |

**Aucun serveur n'est nécessaire pour écrire** : le store par défaut est un
simple fichier SQLite local. Le serveur MLflow ne sert qu'à *relire*. Tant que
`MLFLOW_ENABLED` est faux, rien n'est tracé et la bibliothèque `mlflow` n'est
même pas importée — le comportement du service est strictement inchangé, en
local comme en CI (les tests tournent monitoring éteint).

### Ce qui est tracé

Un **run MLflow par extraction**, succès comme échec, enregistré après l'envoi
au callback pour ne jamais retarder le traitement.

**Métriques** (numériques, suivies dans le temps) :

| Métrique | Ce qu'elle mesure |
| --- | --- |
| `score_confiance` | Confiance globale de l'extraction (0 à 1) |
| `taux_champs_reconnus` | Part des 10 champs extraits avec une confiance ≥ 0.7 |
| `taux_champs_presents` | Part des 10 champs extraits, quelle que soit leur fiabilité |
| `confiance_<champ>` (×10) | Confiance de chaque champ pris isolément |
| `extraction_reussie` | 1 en cas de succès, 0 sur un payload d'échec |
| `duree_secondes` | Durée du pipeline complet |

Le seuil de 0.7 n'est pas arbitraire : c'est la valeur qu'attribue
`confidence.py` à un champ présent et non démenti par un contrôle d'intégrité.
En dessous, le champ est soit absent (0), soit mal formé (0.2), soit invalidé
par un contrôle (0.4). L'écart entre les deux taux distingue « le champ
manque » de « le champ est là mais douteux » — deux problèmes différents.

**Tags** (dimensions de filtrage et de regroupement) :

| Tag | Valeurs | Intérêt |
| --- | --- | --- |
| `id_document` | entier | Retrouver le document derrière un run dégradé |
| `statut` | `succes`, `echec` | Suivre le taux d'échec |
| `type_document` | `facture`, `devis`, `avoir`, `inconnu`, `non_calcule` | Répartition des documents reçus |
| `mode_extraction` | `pdf_natif`, `ocr`, `inconnu` | **Explique** une dérive du score |
| `modele_llm` | nom du modèle Groq | Comparer deux modèles |
| `alerte` | `true`, `false` | Isoler les extractions dégradées |

`mode_extraction` mérite une mention : une baisse du score moyen s'explique
bien plus souvent par « davantage de documents scannés arrivent » que par une
dégradation du modèle. Sans ce tag, on voit la dérive sans pouvoir l'expliquer.

### Données sensibles

**Seuls des agrégats sont tracés.** Le contenu du run est construit depuis une
liste blanche explicite (`src/core/monitoring.py`) : le payload n'est jamais
sérialisé, ses champs ne sont jamais parcourus. Ne partent que des nombres
entre 0 et 1, une durée, des étiquettes à valeurs bornées et l'`id_document`.

Ne sortent **jamais** : le texte brut du document, les SIRET, l'IBAN, le numéro
de facture, les montants, les dates, les désignations de lignes, le nom du
fichier, ni aucun secret. Un test vérifie cette garantie de bout en bout sur ce
que le store a réellement écrit (`tests/test_monitoring.py`).

L'`id_document` est le seul identifiant tracé. Ce n'est pas une donnée
personnelle — un entier interne, qui n'apprend rien sans accès à la base de
l'API data — mais il est indispensable pour retrouver le document derrière un
run à 0.3.

### Restitution

L'interface MLflow, lancée en local :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# puis http://localhost:5000
```

Trois lectures utiles :

- **tableau des runs** : une ligne par extraction, triable par score, filtrable
  par tag — `tags.alerte = 'true'`, `tags.statut = 'echec'`,
  `tags.mode_extraction = 'ocr'` ;
- **graphe temporel** d'une métrique sur l'ensemble des runs : la courbe de
  `score_confiance` dans le temps, c'est-à-dire la dérive ;
- **comparaison de runs** : sélectionner des runs de `modele_llm` différents et
  comparer leurs métriques sur la même population de documents.

Les dix séries `confiance_<champ>` répondent à « quel champ est chroniquement
faible ? », c'est-à-dire à « que faut-il corriger dans le prompt ? ».

### Alerte — compromis assumé

**MLflow ne sait pas alerter.** C'est sa limite face à un couple
Prometheus/Grafana, et elle est assumée : le reste de ce qu'apporte MLflow (la
notion de run, la comparaison de modèles, le suivi par extraction) n'a pas
d'équivalent simple côté métriques, et l'API data couvre déjà l'alerting
applicatif.

À défaut d'alerting natif, un score sous `MONITORING_SEUIL_ALERTE` déclenche
deux choses : un `WARNING` dans les logs applicatifs (récupérable par n'importe
quel collecteur de logs) et le tag `alerte=true` sur le run, qui rend les
extractions dégradées filtrables en un clic dans l'interface. Une vraie règle
d'alerting viendra avec la centralisation des logs.

### Notes d'exploitation

- Le store par défaut est un fichier SQLite dans le répertoire de travail. Le
  store « répertoire de fichiers » (`file:./mlruns`) existe toujours mais
  MLflow l'a placé en mode maintenance : SQLite est le backend local
  recommandé. `MLFLOW_TRACKING_URI` accepte aussi l'URL d'un serveur MLflow.
- Un run par extraction : prévoir une purge périodique du store en cas de fort
  volume.
- Le service n'installe que `mlflow-skinny` (client de tracking). Le paquet
  `mlflow` complet, nécessaire à l'interface, est une dépendance de
  développement.
