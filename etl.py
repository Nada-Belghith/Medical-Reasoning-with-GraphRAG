# -------------------------------------------------
# etl.py - Peuplement du Graphe
# -------------------------------------------------
import json
from langchain_community.graphs import Neo4jGraph
from utils import sanitize

def populate_graph(graph: Neo4jGraph, json_path: str = "medical_data.json"):
    """Lit les données JSON et peuple le graphe Neo4j."""
    print("📥 Lecture du fichier medical_data.json...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            medical_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERREUR : Le fichier {json_path} est manquant.")
        return

    print("🧱 Création du graphe médical dans Neo4j...")
    
    for disease in medical_data:
        disease_name = disease["maladie"]
        disease_id = sanitize(disease_name)

        # Créer maladie
        graph.query(f"""
            MERGE ({disease_id}:Disease {{name: "{disease_name}"}})
        """)

        # Symptômes
        for s in disease.get("symptomes", []):
            s_id = sanitize(s)
            cypher_query = f"""
                MERGE (symptom_{s_id}:Symptom {{name: "{s}"}})
                MERGE (disease_{disease_id}:Disease {{name: "{disease_name}"}})
                MERGE (disease_{disease_id})-[:has_symptom]->(symptom_{s_id})
            """
            graph.query(cypher_query)

        # Traitements
        for t in disease.get("traitements", []):
            t_id = sanitize(t)
            cypher_query = f"""
                MERGE (treatment_{t_id}:Treatment {{name: "{t}"}})
                MERGE (disease_{disease_id}:Disease {{name: "{disease_name}"}})
                MERGE (disease_{disease_id})-[:treated_with]->(treatment_{t_id})
            """
            graph.query(cypher_query)

        # Causes
        for c in disease.get("causes", []):
            c_id = sanitize(c)
            cypher_query = f"""
                MERGE (cause_{c_id}:Cause {{name: "{c}"}})
                MERGE (disease_{disease_id}:Disease {{name: "{disease_name}"}})
                MERGE (disease_{disease_id})-[:caused_by]->(cause_{c_id})
            """
            graph.query(cypher_query)

    print("✅ Graphe médical créé avec succès !")
