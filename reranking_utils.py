from cross_encoder_reranker import CrossEncoderReranker

def get_reranked_results(query, vector_db, feedback_reranker, reranking_strategy="hybrid", k=3, initial_k=10):
    """
    Get reranked search results using various reranking strategies.
    
    Args:
        query (str): The user query
        vector_db: Vector database with similarity_search method
        feedback_reranker: Instance of the feedback-based reranker
        reranking_strategy (str): One of "semantic", "cross_encoder", "feedback", "hybrid", or "ensemble"
        k (int): Number of final results to return
        initial_k (int): Number of initial results to retrieve for reranking
        
    Returns:
        list: The top k reranked documents
    """
    # Initialize cross-encoder reranker
    cross_encoder = CrossEncoderReranker()
    
    # Get initial results
    raw_docs = vector_db.similarity_search(query, k=initial_k)
    print(raw_docs) 
    if not raw_docs:
        return []
    
    # Apply chosen reranking strategy
    if reranking_strategy == "semantic":
        # Just return the vector search results (already ranked by semantic similarity)
        return raw_docs[:k]
        
    elif reranking_strategy == "cross_encoder":
        # Rerank using cross-encoder only
        return cross_encoder.rerank_results(query, raw_docs, k=k)
        
    elif reranking_strategy == "feedback":
        # Rerank using feedback reranker only
        return feedback_reranker.rerank_results(query, raw_docs, k=k)
        
    elif reranking_strategy == "hybrid":
        # Hybrid search combining semantic similarity and cross-encoder
        return cross_encoder.hybrid_search(
            query, 
            raw_docs, 
            vector_db.embedding_function, 
            alpha=0.3,  # Weight more towards cross-encoder
            k=k
        )
        
    elif reranking_strategy == "ensemble":
        # Ensemble approach combining feedback reranker and cross-encoder
        return cross_encoder.combine_with_feedback_reranker(
            query,
            raw_docs,
            feedback_reranker,
            alpha=0.7,  # Weight more towards cross-encoder
            k=k
        )
    
    # Default to hybrid if strategy not recognized
    return cross_encoder.hybrid_search(
        query, 
        raw_docs, 
        vector_db.embedding_function, 
        alpha=0.3,
        k=k
    )
