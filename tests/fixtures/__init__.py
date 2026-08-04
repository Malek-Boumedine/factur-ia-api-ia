"""Jeux de données de test : documents d'exemple et réponses modèle simulées.

Deux modules, un par frontière du pipeline :

- ``documents`` : les fichiers d'entrée (PDF natifs, PDF scannés, images, cas
  corrompus) — ce que reçoit l'endpoint ;
- ``llm`` : les réponses du modèle Groq — ce que le pipeline reçoit en retour de
  la structuration, jamais appelée pour de vrai en test.

Règle absolue : **aucune donnée réelle**. Tous les identifiants sont inventés
(cf. ``documents``) et aucun binaire n'est versionné — tout est généré à la volée.
"""
