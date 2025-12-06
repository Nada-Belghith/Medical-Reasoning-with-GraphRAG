# ===========================================
# agent_chatbot_rag.py (VERSION FINALE STABLE - JSON + correction relations)
# ===========================================

import os
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import ChatOllama
import json
import re

# -------------------------------------------------
# 0️⃣ Fonction utilitaire pour créer des IDs propres
# -------------------------------------------------
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

# -----------------------------
# 1️⃣ Connexion à Neo4j
# -----------------------------
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
)
graph.refresh_schema()

# -----------------------------
# 2️⃣ LLM (Llama3)
# -----------------------------
llm = ChatOllama(model="llama3", temperature=0)

# -----------------------------
# 3️⃣ Peuplement du graphe (JSON)
# -----------------------------
print("📥 Lecture du fichier medical_data.json...")
try:
    with open("medical_data.json", "r", encoding="utf-8") as f:
        medical_data = json.load(f)
except FileNotFoundError:
    print("❌ ERREUR : Le fichier medical_data.json est manquant.")
    exit()

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

# -----------------------------
# 4️⃣ Prompts GraphRAG (force lowercase pour relations)
# -----------------------------
cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
Generate a Cypher query to answer the question using the Neo4j graph.

STRICT RULES:
- Always use relationships exactly as: has_symptom, treated_with, caused_by (lowercase!)
- Use labels: Disease, Symptom, Treatment, Cause.
- Do NOT uppercase relationship types.

Schema:
{schema}

Question:
{question}
"""
)

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant that rewrites Neo4j query results into a clear medical explanation.

Query Results:
{context}

Question:
{question}

Final Answer:
"""
)

# -----------------------------
# 5️⃣ Initialisation GraphRAG
# -----------------------------
qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_generation_prompt,
    qa_prompt=qa_generation_prompt,
    allow_dangerous_requests=True
)

# -----------------------------
# 6️⃣ Fonction pour poser une question
# -----------------------------
def ask_graphrag(question: str):
    try:
        response = qa_chain.invoke({"query": question})
        # Extraire les résultats de la requête Cypher
        result = response.get("result", "")
        if not result or result == "[]":
            return "⚠️ Aucune donnée trouvée dans Neo4j pour cette question."
        return result
    except Exception as e:
        return f"❌ Erreur: {e}"

# -----------------------------
# 7️⃣ Interface console
# -----------------------------
if __name__ == "__main__":
    print("🤖 Agent GraphRAG Médical (Llama3 + Neo4j) prêt !")
    print("📌 Tapez 'exit' pour quitter.\n")
    # Questions de test
    test_questions = [
        "Quels sont les symptômes de la maladie Diabète de type 2 ?",
        "Quels sont les traitements pour l'Asthme ?",
        "Quelles sont les causes de l'Hypertension ?",
    ]
    for tq in test_questions:
        print(f"Question Test > {tq}")
        print(ask_graphrag(tq))

    while True:
        q = input("Question > ")
        if q.lower() == "exit":
            break
        print(ask_graphrag(q))