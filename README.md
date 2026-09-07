# factur-ia-api-ia

Service d'extraction de factures du système **Factur-IA**. Il reçoit un PDF ou une image de l'API data, en extrait le texte (lecture directe pour un PDF natif, reconnaissance optique pour un scan), le structure par un modèle de langage, puis renvoie le résultat par un appel de retour authentifié.

## Démarrage

```bash
git clone https://github.com/Malek-Boumedine/factur-ia-api-ia && cd factur-ia-api-ia

uv sync --all-groups
uv run uvicorn src.main:app --reload --port 8001
```

Copier `.env.example` en `.env` et renseigner au minimum `SECRET_OCR_TOKEN` (partagé avec l'API data — **la même valeur des deux côtés**) et `GROQ_API_KEY` — sans elles, l'application ne démarre pas.

```bash
openssl rand -hex 32    # génère le jeton partagé
```

### Vérifications

```bash
uv run pytest --cov=src            # tests
uv run mypy src/                   # typage strict
uv run pre-commit run --all-files  # lint et formatage
```

### Avec Docker

```bash
docker compose up --build          # API sur http://localhost:8001
```

Le code est monté depuis l'hôte, uvicorn recharge à chaud. Le compose force l'URL de l'API data vers l'hôte — **elle doit écouter sur `0.0.0.0`**, sinon elle reste injoignable depuis le conteneur.

Les **poids de reconnaissance optique** (~98 Mo) ne sont pas dans l'image de développement : ils sont téléchargés au premier document scanné dans un volume qui survit aux redémarrages. Le premier scan est donc lent, et `GET /ready` répond 503 tant que le volume est vide. Pour l'amorcer :

```bash
docker compose run --rm api uv run --no-sync \
  python -c "import easyocr; easyocr.Reader(['fr','en'], gpu=False)"
```

Rien d'autre dans ce compose : pas de base de données, pas de broker, pas de reverse proxy — le service n'en a pas besoin.

> **Image ~1,5 Go.** PyTorch est résolu depuis l'index processeur : les roues standard embarquent CUDA sur Linux, soit environ 4 Go de paquets pour un matériel graphique que la reconnaissance n'utilise jamais.

### Image de production

```bash
docker build -f Dockerfile.prod -t factur-ia-api-ia:prod .
```

Destinée à Cloud Run : code figé, uvicorn sans rechargement, utilisateur non-root, aucun secret embarqué. Les **poids sont cuits dans l'image** (~1,8 Go au total) : rien n'est téléchargé à l'exécution, et `GET /ready` devient un contrôle d'intégrité d'image. Les langues sont figées au moment de la construction.

## Collecte des questions-réponses réglementaires

Un batch indépendant du pipeline collecte des questions-réponses publiques sur la facturation électronique et les agrège dans `data/faq.csv` :

```bash
uv run python -m src.scraping
```

Sources, règles de sélection et limites : voir **[src/scraping/README.md](src/scraping/README.md)**.

## Tests

**266 tests**, exécutés en une dizaine de secondes, **sans aucun réseau** : le modèle de langage, la reconnaissance optique et l'appel de retour sont toujours simulés, et une garde installée dans `tests/conftest.py` fait échouer tout test qui tenterait une connexion réelle.

Les documents d'exemple sont générés en mémoire — aucun binaire versionné, aucune donnée réelle, les identifiants sont inventés mais structurellement valides.

**Couverture** : 100 % sur le périmètre API (extraction, retour, configuration, point d'entrée), 76 % sur l'ensemble de `src/` — l'écart vient de deux modules hors API : le collecteur de questions-réponses et la preuve de concept de recherche sémantique.

La [stratégie de test](https://github.com/Malek-Boumedine/factur-ia-meta/blob/main/docs/strategie-de-test.md) détaille, pour chaque étape du pipeline, le périmètre visé et les limites connues — notamment l'absence de vérité terrain, qui interdit toute mesure du taux d'erreur réel.

## Sondes de disponibilité

Deux routes destinées à la plateforme : publiques (elle sonde sans en-tête d'authentification), mais hors contrat OpenAPI et sans aucune information exploitable — ni version, ni configuration, ni détail d'erreur.

| Route | Rôle | Vérifie | Échec |
|---|---|---|---|
| `GET /health` | Le processus est-il vivant ? | rien | **Redémarrage** du conteneur |
| `GET /ready` | L'instance peut-elle travailler ? | Présence des poids sur disque | **Retrait du trafic**, sans redémarrage |

`/health` répond 200 inconditionnellement, sans aucune entrée-sortie : son échec redémarrant le conteneur, la faire dépendre d'un tiers ferait redémarrer en boucle des instances parfaitement saines.

**La règle appliquée à `/ready`** : ne sortir une instance du trafic que si la panne lui est **locale** et qu'une autre instance ferait mieux. Une panne partagée ne se répare pas en retirant du trafic.

Une seule dépendance satisfait ce critère, les poids de reconnaissance : sans eux, leur téléchargement se déclencherait au milieu du pipeline, et une indisponibilité du dépôt distant se traduirait en extraction ratée pour un document pourtant lisible. Le contrôle est une simple présence de fichier — pas de réseau, environ une milliseconde.

Ne sont volontairement **pas** vérifiés : le fournisseur du modèle (jamais d'appel payant depuis une sonde interrogée en continu, et se retirer du trafic nous priverait d'émettre les échecs proprement) et l'API data (panne partagée : si elle est indisponible, elle ne nous envoie plus rien).

### Configuration de la plateforme

**Le processeur doit rester alloué en continu** : le pipeline tourne en tâche de fond **après** la réponse 202 — en facturation à la requête, il serait étranglé dès la réponse envoyée et l'extraction gelée en plein traitement.

**Concurrence basse**, un seul worker : chaque extraction est gourmande en mémoire (environ 1 Go résident, plus jusqu'à 500 Mo d'images pour un scan long, sur 2 processeurs virtuels et 4 Go). La montée en charge se fait horizontalement.

**Sonde de démarrage** sur `/health`, avec plusieurs tentatives espacées — environ une minute de marge pour un démarrage à froid.

## Livraison continue

À la publication d'une version, le workflow récupère le tag correspondant, construit l'image de production (jamais celle de développement) avec deux étiquettes — version et empreinte du commit, pas de `latest` —, la pousse, déploie la révision par empreinte, puis interroge la sonde de santé avec un jeton d'identité. Un déclenchement manuel permet de redéployer n'importe quelle version, retour arrière compris.

**Partage des responsabilités** : l'infrastructure possède la *forme* du service — configuration, secrets, dimensionnement — et ignore le champ image ; la livraison possède son *contenu*.

**Pas d'étape de migration** : ce service n'a aucun schéma applicatif. Le suivi de modèle crée ses tables tout seul au premier usage.

**Construction longue, sans cache** : l'image fait environ 1,8 Go et télécharge les poids au moment de la construction, soit dix à quinze minutes à froid. Un cache de couches économiserait cinq à huit minutes, mais s'évincerait entre deux versions — complexité écartée.

### Variables GitHub

Aucun secret : la fédération d'identité rend tout stockage confidentiel inutile. Tout va dans les **variables de dépôt**.

| Variable | Contenu |
|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Fournisseur d'identité, sortie Terraform |
| `GCP_DEPLOY_SA` | `github-deployer-api-ia@<projet>.iam.gserviceaccount.com` |
| `GCP_PROJECT_ID` | Identifiant du projet |
| `GCP_REGION` | `europe-west9` |
| `ARTIFACT_REGISTRY_REPO` | `factur-ia` |
| `CLOUD_RUN_SERVICE` | `factur-ia-api-ia` |

Les ressources correspondantes — compte de déploiement, rôles, liaison d'identité — sont décrites dans le dépôt [`factur-ia-infra`](https://github.com/Malek-Boumedine/factur-ia-infra).

## Suivi de la qualité d'extraction

### À quoi ça sert

Chaque extraction produit des signaux de qualité : un score global, une confiance par champ, un type de document. Le suivi les **trace dans le temps** pour répondre à trois questions : la qualité se dégrade-t-elle, quels champs sont chroniquement mal extraits, et un changement de modèle améliore-t-il les résultats ?

C'est du suivi **de modèle**, distinct du monitoring **applicatif** — latence, erreurs — assuré par la stack d'observabilité.

### Activation

Désactivé par défaut. Quatre variables, documentées dans `.env.example` :

| Variable | Défaut | Rôle |
|---|---|---|
| `MLFLOW_ENABLED` | `False` | Interrupteur unique |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Où sont stockées les métriques |
| `MLFLOW_EXPERIMENT_NAME` | `factur-ia-extraction` | Nom de l'expérience |
| `MONITORING_SEUIL_ALERTE` | `0.7` | Score sous lequel une extraction est signalée |

**Aucun serveur n'est nécessaire pour écrire** : le stockage par défaut est un fichier local, le serveur ne sert qu'à relire. Tant que l'interrupteur est fermé, rien n'est tracé et la bibliothèque n'est même pas importée.

**En production**, le stockage pointe vers une base dédiée sur l'instance Cloud SQL déjà provisionnée — le disque des instances étant éphémère, un fichier local serait perdu à chaque recyclage. Volumétrie mesurée : environ 16 Kio par extraction.

### Ce qui est tracé

Un enregistrement par extraction, succès comme échec, écrit **après** l'envoi du résultat pour ne jamais retarder le traitement.

**Métriques** : score de confiance global, taux de champs reconnus, taux de champs présents, confiance de chacun des dix champs principaux, succès ou échec, durée du pipeline.

Le seuil de 0,7 n'est pas arbitraire : c'est la valeur attribuée à un champ présent et non démenti par un contrôle d'intégrité. En dessous, le champ est soit absent, soit mal formé, soit invalidé.

**Étiquettes** : identifiant du document, statut, type de document détecté, mode d'extraction, modèle appelé, dépassement du seuil.

Le **mode d'extraction** mérite une mention : une baisse du score moyen s'explique bien plus souvent par « davantage de documents scannés arrivent » que par une dégradation du modèle. Sans cette étiquette, on voit la dérive sans pouvoir l'expliquer.

### Données sensibles

**Seuls des agrégats sont tracés.** Le contenu est construit depuis une **liste blanche explicite** : le résultat n'est jamais sérialisé, ses champs ne sont jamais parcourus. Ne partent que des nombres, une durée, des étiquettes à valeurs bornées et l'identifiant du document.

Ne sortent **jamais** : le texte du document, les identifiants d'entreprise, les coordonnées bancaires, le numéro de facture, les montants, les dates, ni aucun secret. Un test vérifie cette garantie sur ce que le stockage a réellement écrit.

### Restitution

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# puis http://localhost:5000
```

Trois lectures utiles : le **tableau des extractions**, filtrable par étiquette ; le **graphe temporel** d'une métrique, qui montre la dérive ; et la **comparaison** entre modèles sur la même population de documents. Les dix séries par champ répondent à « quel champ est chroniquement faible ? », donc à « que faut-il corriger dans l'instruction ? ».
