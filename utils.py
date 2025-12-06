# -------------------------------------------------
# utils.py - Fonctions Utilitaires
# -------------------------------------------------
import re

def sanitize(name: str) -> str:
    """Transforme un nom en identifiant Neo4j valide (variables/nœuds)."""
    if not name:
        return "unknown"
    name = name.lower()
    name = re.sub(r"[éèêë]", "e", name)
    name = re.sub(r"[àâä]", "a", name)
    name = re.sub(r"[îï]", "i", name)
    name = re.sub(r"[ôö]", "o", name)
    name = re.sub(r"[ùûü]", "u", name)
    name = re.sub(r"[ç]", "c", name)
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name
