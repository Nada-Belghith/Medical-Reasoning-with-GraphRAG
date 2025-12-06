# -------------------------------------------------
# main.py - Exécution et Interface Principale
# -------------------------------------------------
import os
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama
from etl import populate_graph
from rag_chain import ask_graphrag_with_path_advanced, initialize_rag_chain, ask_graphrag

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
LLM_MODEL = "llama3"

def run_agent():
    """Initialise le graphe, la chaîne RAG et l'interface utilisateur."""
    try:
        # Connexion Neo4j
        print("🔗 Connexion à Neo4j...")
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        graph.refresh_schema()

        # Initialisation LLM
        print("🧠 Initialisation du modèle Llama3...")
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        # Peuplement du Graphe (ETL)
        populate_graph(graph, json_path="medical_data.json")

        # Initialisation de la chaîne RAG
        print("⚙️ Initialisation de la chaîne GraphRAG...")
        qa_chain = initialize_rag_chain(graph, llm)
        
    except Exception as e:
        print(f"❌ ERREUR FATALE D'INITIALISATION : {e}")
        return

    # Interface Console
    print("\n🤖 Agent GraphRAG Médical (Llama3 + Neo4j) prêt !")
    print("📌 Tapez 'exit' pour quitter.\n")

    # Questions de test # test_questions = [ # "Quels sont les symptômes de la maladie Diabète de type 2 ?", # "Quels sont les traitements pour l'Asthme ?", # "Quelles sont les causes de l'Hypertension ?", # ] # for tq in test_questions: # print(f"Question Test > {tq}") # print(ask_graphrag(qa_chain, tq))
    
    # Exemple avec chemin (graph path)
    question = "Quels sont les symptômes de la maladie Diabète de type 2 ?"
    result_with_path = ask_graphrag_with_path_advanced(qa_chain, question)  # On passe qa_chain

    print("Réponse RAG:", result_with_path["answer"])
    print("Chemin parcouru (Graph Path):")
    for step in result_with_path["graph_path"]:
        print(step)

    # Boucle interactive
    while True:
        q = input("Question > ")
        if q.lower() == "exit":
            break
        print(ask_graphrag(qa_chain, q))  # Version classique pour les questions rapides

if __name__ == "__main__":
    run_agent()
