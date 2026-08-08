# Collecte des FAQ sur la facturation électronique

Script d'agrégation **batch** qui collecte des questions-réponses publiques sur la réforme de la facturation électronique et les rassemble dans un CSV unique, `data/faq.csv`. Ce jeu de données alimentera un futur **chatbot RAG** destiné à répondre aux questions réglementaires des utilisateurs de Factur-IA.

## Pourquoi ce module vit ici

Le module est hébergé dans l'API IA parce que le futur RAG y vivra aussi : ils partageront les mêmes ressources (dépendances, conteneur, futur pipeline de vectorisation). Il reste pour autant **totalement indépendant du pipeline d'extraction de factures** : aucun routeur FastAPI, aucun import depuis `extractions/`, `callback/` ou `main.py`, aucune variable de `core/config.py`. On peut lancer la collecte sans clé Groq ni token OCR, et l'API démarre sans que ce module soit jamais importé.

## Sources

### Retenues

| Source | URL | Structure | Volume¹ | Métadonnées fournies |
| --- | --- | --- | --- | --- |
| **DGFiP** (impots.gouv.fr) | [`/professionnel/je-passe-la-facturation-electronique`](https://www.impots.gouv.fr/professionnel/je-passe-la-facturation-electronique) | Une page liste (cartes du Système de Design de l'État), puis **une page par réponse** | 7 Q/R | Date de publication par réponse (`JJ/MM/AAAA`), URL propre à chaque question |
| **Le Coin des Entrepreneurs** | [`/questions-reponses-facturation-electronique/`](https://www.lecoindesentrepreneurs.fr/questions-reponses-facturation-electronique/) | **Page WordPress unique** : chaque question est un titre `h2` ancré, sa réponse est le contenu jusqu'au `h2` suivant | 6 Q/R | Date de publication de la page (ISO 8601 avec fuseau, commune à toutes les Q/R du dossier), URL avec ancre par question |

¹ Volume constaté à la dernière collecte — il évolue avec les sites.

La DGFiP est la source officielle (l'administration qui pilote la réforme) ; Le Coin des Entrepreneurs apporte un angle complémentaire, orienté praticien.

Les `robots.txt` des deux sites ont été vérifiés : les pages collectées ne sont pas interdites aux robots (impots.gouv.fr n'exclut que ses zones techniques Drupal, lecoindesentrepreneurs.fr que son administration WordPress).

### Écartée : economie.gouv.fr

La FAQ d'economie.gouv.fr est protégée par un challenge JavaScript Cloudflare : la page réelle n'est servie qu'à un navigateur complet. La contourner imposerait un navigateur headless (Playwright/Selenium), soit plusieurs centaines de Mo dans une image Docker qui vient d'être ramenée de 6,5 à 1,5 Go. Le coût est disproportionné pour une FAQ dont la matière recoupe largement celle de la DGFiP — source écartée.

## Pourquoi du scraping HTML et pas une API

Légifrance expose une API officielle (PISTE), mais elle sert les textes juridiques bruts — pas les FAQ pédagogiques, qui sont précisément le format recherché pour un chatbot de questions-réponses. Ni impots.gouv.fr ni Le Coin des Entrepreneurs n'exposent d'API pour ces contenus : le scraping HTML est le seul moyen de les collecter.

Il faut être honnête sur le compromis : **un scraper est plus fragile qu'un contrat d'API**. Les sélecteurs CSS reposent sur la mise en page du moment ; une refonte du site les casse sans préavis. Le module compense en échouant **explicitement et bruyamment** (cf. [Gestion des erreurs](#gestion-des-erreurs)) plutôt qu'en collectant silencieusement une liste vide ou partielle.

## Enchaînement logique de l'algorithme

`python -m src.scraping` exécute `__main__.py`, qui déroule :

1. **Itération sur les scrapers** déclarés dans la liste `SCRAPERS` (`dgfip.py`, puis `lecoindesentrepreneurs.py`). Chaque scraper est appelé dans un `try/except ScrapingError` : l'échec d'une source est compté et journalisé sur stderr, **sans faire tomber les autres**.

2. **Scraper DGFiP** (`dgfip.py`) :
   - télécharge la page liste et sélectionne les cartes de questions (`h3.fr-card__title > a[href^="/professionnel/questions/"]`) ;
   - ignore les cartes en double (certaines sont rendues deux fois pour des variantes d'affichage) ;
   - pour chaque question : **pause de 1 s**, téléchargement de la page réponse, extraction du corps (paragraphes et éléments de listes, en écartant le widget d'avis « Cet article vous a-t-il été utile ? ») et de la date de publication machine-readable (`data-published-date`).

3. **Scraper Le Coin des Entrepreneurs** (`lecoindesentrepreneurs.py`) :
   - télécharge la page unique du dossier, localise le corps de l'article (`div.entry`) et ses titres de questions (`h2.wp-block-heading`) ;
   - pour chaque titre, rassemble la réponse : les blocs qui le suivent jusqu'au `h2` suivant, en ne retenant que les paragraphes (`p.wp-block-paragraph`) et listes (`ul.wp-block-list`) éditoriaux — les encarts publicitaires et boîtes à outils intercalés sont ainsi ignorés d'office ;
   - lit la date de publication dans les métadonnées Open Graph (`article:published_time`).

4. **Agrégation** : chaque scraper renvoie une `list[FaqEntry]` (le `TypedDict` de `common.py` — question, réponse, source, date de parution, URL) ; l'orchestrateur concatène les listes.

5. **Déduplication et écriture** (`storage.py`) : le CSV existant est relu, les entrées déjà connues sont écartées, les nouveautés sont **ajoutées en fin de fichier** avec un horodatage de collecte commun au lot (cf. [Déduplication](#déduplication)).

6. **Compte rendu** sur stdout — c'est ce qu'un futur cron enverra par mail — puis code de sortie : `0` si au moins une source a fonctionné, `1` si toutes ont échoué.

## Nettoyage et homogénéisation des formats

Deux structures HTML très différentes (site Drupal de l'État d'un côté, article WordPress de l'autre) convergent vers un format unique. Les règles :

- **Normalisation du texte** (`clean_text` dans `common.py`) : toutes les suites d'espaces — espaces insécables comprises — sont réduites à une espace simple. Les entités HTML sont décodées par BeautifulSoup en amont. Les apostrophes et guillemets typographiques (`’`, `«  »`) sont **conservés** : ils sont valides en français, les « corriger » dégraderait le texte.
- **Réponses en texte brut structuré** : une réponse est la concaténation de ses paragraphes, un par ligne. Les éléments de liste sont **préfixés d'un tiret** (`- `) : la structure d'énumération reste lisible sans balisage HTML.
- **Unification des dates** : la DGFiP publie `JJ/MM/AAAA` (sans heure), Le Coin des Entrepreneurs une date ISO 8601 avec heure et fuseau. Chacune est parsée dans son format d'origine puis **sérialisée en ISO 8601** dans le CSV — la précision d'origine est conservée (la date DGFiP n'a pas d'heure, celle du Coin des Entrepreneurs en a une). Une date absente donne un champ vide, jamais une valeur inventée.
- **Champ `source` réduit à l'organisme** (`DGFiP`, `LeCoinDesEntrepreneurs`) : c'est l'attribution que le futur chatbot devra citer ; l'URL précise vit dans sa propre colonne.

## Déduplication

Le CSV est **sa propre mémoire** : aucune base annexe. À chaque collecte, `storage.py` relit le fichier et calcule l'empreinte de chaque ligne existante ; seules les entrées inconnues sont ajoutées.

L'empreinte porte sur le couple **(question, réponse)** — pas sur la question seule — pour capter deux types de nouveautés :

- une **question nouvelle** publiée par la source ;
- une **réponse modifiée** par l'administration sur une question existante : la doctrine évolue (calendrier, seuils…), et une réponse corrigée est une information neuve qui doit entrer dans le jeu de données.

Conséquence assumée : une réponse modifiée produit une **nouvelle ligne**, à côté de l'ancienne — le CSV garde l'historique, chaque version portant sa date de collecte. C'est en aval (vectorisation) que la version périmée devra être remplacée (cf. [Perspectives](#perspectives)).

La déduplication opère aussi **au sein d'un même lot** (une entrée ajoutée enrichit immédiatement l'ensemble des empreintes connues), et le scraper DGFiP écarte en amont les cartes de questions rendues en double sur la page liste.

## Gestion des erreurs

Deux exceptions, définies dans `common.py`, distinguent les deux façons dont un scraper peut échouer :

| Exception | Cause | Ce qu'elle signifie |
| --- | --- | --- |
| `SourceUnavailableError` | Timeout, erreur réseau, code HTTP d'erreur | La source est **injoignable** — problème transitoire probable, réessayer plus tard |
| `UnexpectedStructureError` | Un sélecteur ne trouve rien, une réponse est vide, une date est illisible | Le site a **changé de mise en page** — le scraper doit être adapté |

Les deux héritent de `ScrapingError`. Le message d'une `UnexpectedStructureError` précise toujours **quel sélecteur** a échoué et **sur quelle URL**, pour rendre le diagnostic immédiat.

Principes appliqués :

- **Jamais de liste partielle silencieuse** : au sein d'une source, la première page illisible interrompt la collecte de cette source (fail-fast). Une FAQ à moitié collectée sans alerte serait pire qu'un échec franc.
- **Isolation par source** : l'orchestrateur rattrape `ScrapingError` autour de chaque scraper — une source en panne ne fait pas perdre les autres. Toute autre exception est un bug et remonte telle quelle.
- **Code de sortie** : `0` si au moins une source a abouti (les entrées collectées sont sauvegardées), `1` si toutes ont échoué (rien n'est écrit). C'est le signal qu'exploitera l'ordonnanceur.

## Dépendances et exécution

Le module n'utilise que deux bibliothèques : **httpx** (déjà dépendance du service, pour le callback) et **beautifulsoup4** (parsing HTML, seule dépendance ajoutée pour la collecte). Pas de navigateur headless, pas de `requests` en doublon d'httpx.

```bash
uv sync                        # si ce n'est pas déjà fait
uv run python -m src.scraping
```

Exemple de sortie (première collecte) :

```text
DGFiP : 7 entrées collectées
LeCoinDesEntrepreneurs : 6 entrées collectées
Sauvegarde : 13 nouvelles entrées, 0 déjà connues → /chemin/du/projet/data/faq.csv
```

Relancé aussitôt, le script ne réécrit rien :

```text
DGFiP : 7 entrées collectées
LeCoinDesEntrepreneurs : 6 entrées collectées
Sauvegarde : 0 nouvelles entrées, 13 déjà connues → /chemin/du/projet/data/faq.csv
```

Une source en échec est signalée sur stderr sans bloquer le reste :

```text
DGFiP : ERREUR — Réponse HTTP 503 pour https://www.impots.gouv.fr/professionnel/je-passe-la-facturation-electronique
LeCoinDesEntrepreneurs : 6 entrées collectées
Sauvegarde : 0 nouvelles entrées, 6 déjà connues → /chemin/du/projet/data/faq.csv
```

## Format de sortie

Un seul fichier, `data/faq.csv` (UTF-8, en-tête sur la première ligne), six colonnes :

| Colonne | Contenu |
| --- | --- |
| `question` | La question, nettoyée |
| `reponse` | La réponse en texte brut, un paragraphe par ligne, listes préfixées de `- ` |
| `source` | L'organisme : `DGFiP` ou `LeCoinDesEntrepreneurs` |
| `date_heure_parution` | Date de publication annoncée par la source (ISO 8601), vide si non affichée |
| `url` | URL de la réponse (page dédiée côté DGFiP, ancre `#question-N` côté Le Coin des Entrepreneurs) |
| `date_heure_scraping` | Horodatage de la collecte (ISO 8601 avec fuseau), commun à tout le lot |

Le chemin est ancré sur la racine du projet (calculé depuis le fichier source, pas depuis le répertoire courant) : lancé par cron depuis n'importe où, le script écrit toujours au même endroit.

Le répertoire `data/` est **exclu du versionnage** (`.gitignore`) : c'est une donnée collectée, reproductible en relançant le script, qui évoluerait à chaque collecte — sa place n'est pas dans l'historique git du code.

## Limites connues

- **Fragilité des sélecteurs CSS** : une refonte de l'un des sites casse son scraper. C'est structurel au scraping ; le choix fait ici est l'**erreur explicite** (`UnexpectedStructureError` nommant le sélecteur et l'URL) plutôt qu'une collecte silencieusement vide qui passerait inaperçue.
- **Volume modeste** : une dizaine de questions-réponses aujourd'hui. Le jeu de données grossira avec les publications des sources et l'ajout d'autres sources.
- **Une seule source publique officielle** (DGFiP) : economie.gouv.fr étant inaccessible sans navigateur headless, la parole de l'administration ne vient que d'impots.gouv.fr.
- **Perte du balisage HTML** : liens hypertextes, tableaux, gras et titres intermédiaires des réponses sont aplatis en texte brut (seules les listes gardent une trace, via le préfixe `- `).
- **Dates hétérogènes en précision** : jour seul côté DGFiP, horodatage complet côté Le Coin des Entrepreneurs — et pour ce dernier, la date est celle de la page entière, pas de chaque question.

### Politesse

Le scraper s'identifie et ménage les sites consultés :

- **User-Agent identifiable** (`factur-ia-scraper/0.1` avec une adresse de contact) : la source peut savoir qui la consulte et écrire si besoin ;
- **pause de 1 s entre deux requêtes** vers impots.gouv.fr (la seule source nécessitant plusieurs requêtes) ;
- **timeout de 10 s** : pas de connexion qui s'éternise ;
- volume par collecte très faible (une quinzaine de requêtes au total).

## Perspectives

- **Planification** : lancement périodique par cron ; le compte rendu stdout est conçu pour être envoyé par mail, et le code de sortie pour alerter.
- **Vectorisation** : le CSV a vocation à être découpé et vectorisé dans la base du futur chatbot RAG, avec `source` et `url` conservées pour que le chatbot cite ses sources.
- **Remplacement des versions périmées** : le CSV conserve l'ancienne et la nouvelle version d'une réponse modifiée. À la vectorisation, la version récente devra **remplacer** l'ancienne (et non coexister avec elle), sans quoi le chatbot pourrait citer une doctrine périmée — la question et la date de collecte permettent de repérer les versions successives.
