# CHANGELOG

<!-- version list -->

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
