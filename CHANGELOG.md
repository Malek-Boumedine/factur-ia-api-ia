# CHANGELOG

<!-- version list -->

## v1.6.0 (2026-08-31)

### Bug Fixes

- **callback**: Authentification IAM du retour vers l'API data
  ([`dfbf7be`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/dfbf7bec361b3815b40c085de36503ce1a59d032))

- **recherche**: Versionne le corpus de la recherche sémantique
  ([`85478f5`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/85478f51a168ed84bf08a7a1ee8e4d616e24ae5b))

### Features

- **ci**: Chaîne de livraison continue vers Cloud Run
  ([`d5dd25c`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/d5dd25c038e5ac760e5d6e7fda042e8b9803b4ab))

- **docker**: Image de production Cloud Run avec poids EasyOCR embarqués
  ([`4ebfe9d`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/4ebfe9d435c773e4d3acd87304ccde0eee83b4db))

- **monitoring**: Métriques Prometheus de qualité d'extraction sur /metrics
  ([`1aa9c4e`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/1aa9c4ef134135421936a52cc3346248a41fcf96))

- **recherche**: Preuve de concept de recherche sémantique sur le corpus réglementaire
  ([`1f52589`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/1f5258966ba8815c18d89cca22ce7f12e8c40af0))


## v1.5.0 (2026-08-08)

### Bug Fixes

- **config**: Refuse une valeur vide pour les variables requises
  ([`5ae1e1c`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/5ae1e1c0735a32da9ef57a3dea643045410e9fe3))

- **extractions**: Remplace la constante 413 dépréciée par Starlette
  ([`8d5ccc0`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/8d5ccc0a01be9232f39ac15fe94e0fece4971351))

### Documentation

- Reformatage des fichiers README général et README du module scraping
  ([`eeb0288`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/eeb028857b6681d0448c29db80cc909456ea3555))

### Features

- **extractions**: Expose la confiance par champ et le type de document dans le callback
  ([`6d1b400`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/6d1b400ad0e98f68e88f7faa79ee423fc952521b))

- **extractions**: Extrait la date d'échéance des factures
  ([`b6ac91e`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/b6ac91e557874d717a72cc51d98593d3a8fb05a0))

- **infra**: Conteneurise l'API IA pour le développement local
  ([`d9d2c9d`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/d9d2c9d64e7c0bdcfc21b390a990f6a7f62c497b))

- **infra**: Sondes de disponibilité /health et /ready pour Cloud Run
  ([`308d16c`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/308d16c3bf27450d38f8d42d638107e96ef403a9))

- **monitoring**: Suit la qualité des extractions avec MLflow
  ([`3345a30`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/3345a30e13439ef90bedc6d814e774417b9038fe))

- **scraping**: Collecte des FAQ officielles sur la facturation électronique
  ([`0d04c85`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/0d04c85fcc0e286008b58c57216c67950cb244cb))

- **scraping**: Orchestration de la collecte et sauvegarde du jeu de données
  ([`7f10564`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/7f1056455c2c964cd4fb99ac84058db65ce412fd))

### Testing

- **extractions**: Couvre le pipeline de bout en bout sur documents d'exemple
  ([`7f22c86`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/7f22c8605c03af0a024a433b61fba31d288b723f))


## v1.4.0 (2026-07-27)

### Bug Fixes

- **extractions**: Corrige la lecture des montants au format français
  ([`26108b4`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/26108b4532b3f97b228b4cbc383f1d659fc2d6a2))

### Features

- **extractions**: Détecte l'incohérence entre les lignes et le total HT dans le score de confiance
  ([`435c506`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/435c50676a83f2a7de5b4c3a588f054e13fa87bc))


## v1.3.0 (2026-07-16)

### Features

- **callback**: Renvoi du résultat d'extraction à l'API data
  ([`2a9b119`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/2a9b119ef8654e8db41bb695106f416bb22790a7))

- **extractions**: Assemble le pipeline complet d'extraction
  ([`faaa1c5`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/faaa1c53880716791cec55b30994a5a35e77a70d))

- **extractions**: Fiabilise la gestion d'erreurs de bout en bout du pipeline
  ([`651085e`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/651085ee72f265cea57739d045541c1a106c92c4))


## v1.2.0 (2026-07-15)

### Features

- **extractions**: Client LLM Groq pour la structuration
  ([`fc424f7`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/fc424f79c28324ea581abc8fac6d228257f445be))

- **extractions**: Détection du type de document (devis/facture/avoir)
  ([`f5c4a31`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/f5c4a318b90f344518f21f727fbaa625869d989a))

- **extractions**: Score de confiance déterministe de l'extraction
  ([`c1e3591`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/c1e3591dd14538bfedc6828d35fafad081930a78))

- **extractions**: Structuration LLM du texte en données de facture
  ([`5f2f1cc`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/5f2f1cc5e8dcfbd1f006c051353a3bd822bd5c08))

- **extractions**: Validation Pydantic et gestion des extractions inexploitables
  ([`87616f0`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/87616f0ad914670d17051674100876485efef413))


## v1.1.0 (2026-07-06)

### Features

- **extractions**: Détection PDF natif vs scanné
  ([`225e27d`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/225e27d7b5a73422a2946af90efe6373c0c14796))

- **extractions**: Extraction de texte par OCR (images et PDF scannés)
  ([`f4a89f7`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/f4a89f7d1ee14a4c79fe89f91a32e4c7657085c4))

- **extractions**: Extraction du texte des PDF natifs
  ([`7f689db`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/7f689db5767dcf196dea71f2185cb21b267a86e6))

- **extractions**: Réception des documents via POST /extractions
  ([`f1994cf`](https://github.com/Malek-Boumedine/factur-ia-api-ia/commit/f1994cf5e5cfd4360c669972d287281a56e88f8d))


## v1.0.0 (2026-07-05)

- Initial Release
