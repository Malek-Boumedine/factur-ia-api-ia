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

### Avec Docker (local)

```bash
docker compose up --build                     # API sur http://localhost:8001
```

Le code est monté depuis l'hôte : uvicorn recharge à chaud. Le `.env` est lu tel
quel, à une exception près — le compose force `DATA_API_BASE_URL` sur
`http://host.docker.internal:8080`, puisque l'API data tourne sur l'hôte et non
dans ce compose. **Elle doit écouter sur `0.0.0.0`** : sur `127.0.0.1`, elle
reste injoignable depuis le conteneur.

Les **poids EasyOCR** (~98 Mo) ne sont pas cuits dans l'image : ils sont
téléchargés au premier document scanné dans le volume `ocr_models`, qui survit
aux redémarrages comme aux reconstructions. Le premier scan est donc lent, et
`GET /ready` répond 503 tant que le volume est vide — c'est exactement ce que la
sonde est censée signaler. Pour l'amorcer et passer au vert tout de suite :

```bash
docker compose run --rm api uv run --no-sync \
  python -c "import easyocr; easyocr.Reader(['fr','en'], gpu=False)"
```

Rien d'autre dans ce compose : pas de base de données (le service n'en a pas),
pas de broker ni de worker (l'asynchrone tient dans le processus), pas de reverse
proxy. Le monitoring MLflow écrit son `mlflow.db` dans le projet monté, sans
volume dédié.

> **Image ~1,5 Go.** `torch` est résolu depuis l'index CPU de PyTorch
> (`tool.uv.sources` dans `pyproject.toml`) : les roues PyPI embarquent CUDA sur
> Linux, soit ~4 Go de paquets `nvidia-*` pour un GPU que l'OCR n'utilise jamais
> (`EASYOCR_GPU=False`). Cette redirection profite aussi au venv local et à la CI.

## Tests

216 tests, 100 % de couverture de `src/`. La suite tourne en une dizaine de
secondes, **sans réseau** : le LLM Groq, EasyOCR et le callback de l'API data
sont toujours simulés, et une garde installée dans `tests/conftest.py` fait
échouer tout test qui tenterait une connexion réelle. Les documents d'exemple
sont générés en mémoire (aucun binaire versionné, aucune donnée réelle : les
SIRET et IBAN sont inventés).

La **[stratégie de test](docs/strategie-de-test.md)** détaille, pour chaque étape
du pipeline, la partie visée, le périmètre, l'approche retenue et les limites
connues — notamment l'absence de vérité terrain, qui interdit toute mesure du
taux d'erreur d'extraction.

## Sondes de disponibilité

Deux routes destinées à la plateforme de déploiement : publiques (Cloud Run
sonde sans en-tête d'authentification), mais **hors contrat OpenAPI** et sans
aucune information exploitable dans les réponses — ni version, ni configuration,
ni détail d'erreur.

| Route | Rôle | Vérifie | Échec |
| --- | --- | --- | --- |
| `GET /health` | Liveness — le processus est-il vivant ? | rien | **redémarrage** du conteneur |
| `GET /ready` | Readiness — cette instance peut-elle mener une extraction à bien ? | poids EasyOCR présents sur disque | **retrait du trafic**, sans redémarrage |

`/health` répond 200 inconditionnellement, sans la moindre I/O : puisque son
échec redémarre le conteneur, la faire dépendre d'un tiers ferait redémarrer en
boucle des instances parfaitement saines.

### Ce que `/ready` vérifie, et pourquoi si peu

La règle appliquée : **ne sortir une instance du trafic que si la panne lui est
locale et qu'une autre instance ferait mieux.** Une panne partagée ne se répare
pas en retirant du trafic — elle se gère par retries et par un échec propre.

Une seule dépendance satisfait ce critère : les **poids EasyOCR**. Sans eux, le
premier document scanné déclencherait leur téléchargement (~98 Mo) *au milieu*
du pipeline ; sur un système de fichiers éphémère, cette attente non bornée se
rejoue à chaque instance froide, et une indisponibilité du CDN se traduirait en
extraction ratée (`score_confiance = 0`) pour un document pourtant lisible. Le
contrôle est une simple présence de fichier : pas de réseau, pas de chargement
de torch, ~1 ms.

Ne sont volontairement **pas** vérifiés :

- **Groq** — jamais d'appel à un service payant depuis une sonde interrogée en
  continu. Surtout, se retirer du trafic parce que Groq est tombé nous priverait
  d'émettre les payloads d'échec : les documents resteraient bloqués « en
  attente » côté API data au lieu de passer proprement en « erreur ». La
  présence de la clé n'est pas testée non plus — `GROQ_API_KEY` est requise par
  la configuration, donc l'application ne démarre pas sans elle.
- **l'API data** (destination du callback) — panne partagée, non locale. Le
  callback a ses propres retries, il intervient en fin de pipeline et non à
  l'entrée, et si l'API data est indisponible elle ne nous envoie plus rien : il
  n'y a aucun trafic à retirer.

Limite assumée : une instance sans poids est retirée du trafic alors qu'elle
traiterait encore les PDF natifs (le cas majoritaire). En production, les poids
seront cuits dans l'image — le contrôle devient alors un contrôle d'intégrité
d'image, toujours vert, rouge immédiatement si l'image est cassée.

### Configuration Cloud Run

- **Startup probe** sur `/health` : `periodSeconds: 10`, `failureThreshold: 6`,
  `timeoutSeconds: 4` — laisse ~60 s de démarrage à froid.
- **Liveness probe** sur `/health` : `periodSeconds: 30`, `timeoutSeconds: 4`,
  `failureThreshold: 3`.
- Cloud Run ne propose pas de readiness probe continue au sens Kubernetes :
  `/ready` sert d'*uptime check* (une alerte « instance hors trafic ») et de
  readiness le jour où le service tournerait sur GKE.
- Le répertoire des poids EasyOCR se pilote par `EASYOCR_MODULE_PATH` (défaut
  `~/.EasyOCR/`), utile pour pointer un volume ou l'emplacement choisi dans
  l'image.

> Si une instrumentation HTTP (OpenTelemetry, Prometheus) est ajoutée plus tard,
> **exclure ces deux routes** : sondées en continu, elles écraseraient les
> statistiques de latence et de taux d'erreur du trafic réel.

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
