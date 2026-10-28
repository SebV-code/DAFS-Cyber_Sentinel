# Changelog

## [1.0] - 2026-10-28

### pipeline/data.py — chargement des données (Roland)

**Corrigé**
- Le pipeline plantait au démarrage avec pandas 3 : la colonne cible n'était plus reconnue comme du texte, donc plus convertie en 0/1. La détection utilise désormais `pd.api.types.is_string_dtype`.

**Ajouté**
- Journalisation (`logging`) : nombre de lignes et de colonnes chargées, répartition de la cible (connexions normales / anomalies).
- Contrôles bloquants du schéma : le pipeline s'arrête avec un message clair si le fichier n'a pas 42 colonnes ou si une colonne numérique contient autre chose que des nombres.
- Alertes non bloquantes : lignes en double et colonnes constantes. `num_outbound_cmds` et `is_host_login` ont une seule valeur, à retirer éventuellement au prétraitement.
- Règle des valeurs manquantes : signalement des colonnes concernées, suppression des lignes sans cible, valeurs catégorielles manquantes remplacées par `"unknown"`. Les valeurs numériques manquantes sont laissées à l'étape de prétraitement.
- Typage explicite des colonnes catégorielles (`protocol_type`, `service`, `flag`) en texte dès la lecture du CSV.

**Inchangé**
- La signature de `load_data()` et son résultat. Les métriques du modèle sont identiques (accuracy ≈ 0,997).