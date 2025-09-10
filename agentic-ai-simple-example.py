from sentence_transformers import SentenceTransformer, util

# 1. Load a local sentence-transformer model (fast, small one)
# You can swap with "all-MiniLM-L6-v2" or any local HuggingFace model you’ve downloaded.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Build a simple knowledge base
knowledge_base = [
    "Databricks Delta tables support ACID transactions.",
    "ZORDER in Databricks helps improve query performance by colocating related information.",
    "The MS Graph API uses OAuth2 scopes (scp values) for delegated permissions.",
    "Doc2Vec is available in Python via gensim, but Julia uses TextAnalysis.jl instead.",
]

# Precompute embeddings
kb_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

# 3. Define the simple agent
def agent_loop():
    print("🤖 Simple Knowledge Agent (type 'quit' to exit)")
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit"]:
            break
        
        # Encode user query
        q_emb = model.encode(query, convert_to_tensor=True)
        
        # Find the best matching knowledge
        hits = util.semantic_search(q_emb, kb_embeddings, top_k=1)
        best_match = hits[0][0]
        answer = knowledge_base[best_match["corpus_id"]]
        
        # Agent "thinks" and "acts"
        print(f"Agent: Based on my knowledge, {answer}\n")

# 4. Run it
if __name__ == "__main__":
    agent_loop()
