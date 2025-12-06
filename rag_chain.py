# -------------------------------------------------
# rag_chain.py - Logique RAG et Prompts
# -------------------------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import ChatOllama

def initialize_rag_chain(graph: Neo4jGraph, llm: ChatOllama) -> GraphCypherQAChain:
    """Initialise et retourne la chaîne GraphCypherQAChain."""
    
    # Prompts de génération (Cypher et QA)
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

    # Initialisation de la chaîne
    qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_generation_prompt,
        qa_prompt=qa_generation_prompt,
        allow_dangerous_requests=True
    )
    return qa_chain

def ask_graphrag(qa_chain: GraphCypherQAChain, question: str) -> str:
    """Pose une question et retourne la réponse."""
    try:
        response = qa_chain.invoke({"query": question})
        result = response.get("result", "")
        if not result or result == "[]":
            return "⚠️ Aucune donnée trouvée dans Neo4j pour cette question."
        return result
    except Exception as e:
        return f"❌ Erreur: {e}"
def ask_graphrag_with_path_advanced(qa_chain, question: str):
    """
    Pose une question au GraphRAG et retourne :
    - la réponse textuelle générée
    - le chemin complet parcouru dans le graphe avec profondeur et score
    """
    response = qa_chain.invoke({"query": question})

    # Extraire le vrai résultat du Cypher (liste de dicts)
    raw_result = response.get("raw_query_result")  # ou "data" selon ta version
    if raw_result is None:
        raw_result = []

    graph_path = []
    
    # On va simuler un score et profondeur pour chaque lien
    for row in raw_result:
        disease_name = row.get("Disease") or row.get("d.name") or "Unknown Disease"

        # Profondeur 1 = directement lié à la maladie
        depth = 1  

        if "Symptom" in row or "s.name" in row:
            symptom_name = row.get("Symptom") or row.get("s.name")
            graph_path.append({
                "node": f"Disease: {disease_name}",
                "relation": "has_symptom",
                "next_node": f"Symptom: {symptom_name}",
                "depth": depth,
                "score": 1.0  # Tu peux remplacer par un score de similarité réel
            })
        if "Treatment" in row or "t.name" in row:
            treatment_name = row.get("Treatment") or row.get("t.name")
            graph_path.append({
                "node": f"Disease: {disease_name}",
                "relation": "treated_with",
                "next_node": f"Treatment: {treatment_name}",
                "depth": depth,
                "score": 1.0
            })
        if "Cause" in row or "c.name" in row:
            cause_name = row.get("Cause") or row.get("c.name")
            graph_path.append({
                "node": f"Disease: {disease_name}",
                "relation": "caused_by",
                "next_node": f"Cause: {cause_name}",
                "depth": depth,
                "score": 1.0
            })

    # Réponse finale générée pour l'utilisateur
    final_answer = response.get("result", "⚠️ Pas de réponse générée.")

    return {
        "answer": final_answer,
        "graph_path": graph_path
    }
