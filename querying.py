# Load the stored database
vector_db = FAISS.load_local("book_index", embeddings)

def query_book(query):
    """Retrieve the most relevant book section for a given query."""
    results = vector_db.similarity_search(query, k=3)  # Top 3 results
    return [r.page_content for r in results]

# Example query
query = "What happened last time?"
results = query_book(query)

for idx, res in enumerate(results):
    print(f"Result {idx+1}:\n{res}\n")
