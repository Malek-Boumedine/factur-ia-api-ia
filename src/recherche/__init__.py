"""POC de recherche sémantique sur la FAQ réglementaire collectée par le scraping.

Module indépendant du pipeline d'extraction et du code du scraping : son seul
point de contact avec le reste du projet est le fichier ``data/faq.csv``
(chemin et colonnes), jamais un import. Dépendances dans le groupe ``poc`` du
``pyproject.toml`` — jamais installées dans l'image de production.
"""
