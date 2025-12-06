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
        return_intermediate_steps=True,
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
    import re
    
    try:
        # Invoke la chaîne avec return_intermediate_steps=True
        response = qa_chain.invoke({"query": question})
        
        # Récupérer la réponse finale pour l'utilisateur
        final_answer = response.get("result", "⚠️ Pas de réponse générée.")
        
        # Récupérer les étapes intermédiaires
        intermediate_steps = response.get("intermediate_steps", [])
        
        graph_path = []
        disease_name = "Unknown Disease"
        
        # Traiter chaque étape intermédiaire
        for step in intermediate_steps:
            # Chercher la requête générée pour extraire le nom de la maladie
            if "query" in step:
                query_text = str(step.get("query", ""))
                # Pattern pour extraire le nom de la maladie du Cypher
                # Cherche: {name: "..." } ou = "..."
                matches = re.findall(r'(?:name:\s*"([^"]+)"|=\s*"([^"]+)")', query_text)
                if matches:
                    # Prendre le premier match qui n'est pas vide
                    for match_tuple in matches:
                        potential_name = match_tuple[0] or match_tuple[1]
                        if potential_name:
                            disease_name = potential_name
                            break
            
            # C'est le résultat de la requête Cypher
            if "context" in step:
                context_data = step.get("context", [])
                
                for row in context_data:
                    # Essayer différentes clés possibles
                    row_disease = (
                        row.get("Disease") or 
                        row.get("d.name") or 
                        row.get("d") or 
                        disease_name
                    )
                    if row_disease and row_disease != "Unknown Disease":
                        disease_name = row_disease
                    
                    # Vérifier pour les symptômes
                    symptom_keys = ["Symptom", "SymptomName", "s.name", "symptom_name"]
                    symptom_name = next((row.get(k) for k in symptom_keys if k in row), None)
                    if symptom_name:
                        graph_path.append({
                            "node": f"Disease: {disease_name}",
                            "relation": "has_symptom",
                            "next_node": f"Symptom: {symptom_name}",
                            "depth": 1,
                            "score": 0.95
                        })
                    
                    # Vérifier pour les traitements
                    treatment_keys = ["Treatment", "t.name", "treatment_name"]
                    treatment_name = next((row.get(k) for k in treatment_keys if k in row), None)
                    if treatment_name:
                        graph_path.append({
                            "node": f"Disease: {disease_name}",
                            "relation": "treated_with",
                            "next_node": f"Treatment: {treatment_name}",
                            "depth": 1,
                            "score": 0.95
                        })
                    
                    # Vérifier pour les causes
                    cause_keys = ["Cause", "cause_name", "c.name"]
                    cause_name = next((row.get(k) for k in cause_keys if k in row), None)
                    if cause_name:
                        graph_path.append({
                            "node": f"Disease: {disease_name}",
                            "relation": "caused_by",
                            "next_node": f"Cause: {cause_name}",
                            "depth": 1,
                            "score": 0.95
                        })
        
        # Si pas de données extraites, afficher le message par défaut
        if not graph_path:
            graph_path.append({
                "node": "Query Executed",
                "relation": "---",
                "next_node": "No intermediate paths found",
                "depth": 0,
                "score": 0.0
            })
        
        return {
            "answer": final_answer,
            "graph_path": graph_path
        }
    except Exception as e:
        import traceback
        return {
            "answer": f"❌ Erreur lors du traitement : {e}",
            "graph_path": [{"node": "Error", "relation": "---", "next_node": str(traceback.format_exc()), "depth": 0, "score": 0.0}]
    }
