import numpy as np
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """A reranker that uses a cross-encoder model to improve search results."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name (str): The name of the cross-encoder model to use.
                Default is "cross-encoder/ms-marco-MiniLM-L-6-v2"
        """
        self.model_name = model_name
        self.model = None  # Lazy loading to avoid unnecessary model loading
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if self.model is None:
            self.model = CrossEncoder(self.model_name)
    
    def rerank_results(self, query, docs, k=3):
        """
        Rerank search results using the cross-encoder model.
        
        Args:
            query (str): The user query
            docs (list): List of document objects with a page_content attribute
            k (int): Number of top results to return
            
        Returns:
            list: The top k reranked documents
        """
        self._ensure_model_loaded()
        
        if not docs:
            return []
        
        # Create query-document pairs for scoring
        query_doc_pairs = [[query, doc.page_content] for doc in docs]
        
        # Score all pairs
        scores = self.model.predict(query_doc_pairs)
        
        # Pair scores with documents and sort by score (descending)
        scored_docs = list(zip(scores, docs))
        reranked_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        
        return reranked_docs[:k]
    
    def combine_with_feedback_reranker(self, query, docs, feedback_reranker, alpha=0.7, k=3):
        """
        Combine scores from cross-encoder and feedback-based reranker.
        
        Args:
            query (str): The user query
            docs (list): List of document objects
            feedback_reranker: An instance of your feedback-based reranker
            alpha (float): Weight for cross-encoder scores (1-alpha for feedback scores)
            k (int): Number of top results to return
            
        Returns:
            list: The top k documents ranked by combined score
        """
        self._ensure_model_loaded()
        
        if not docs:
            return []
        
        # Get scores from cross-encoder
        query_doc_pairs = [[query, doc.page_content] for doc in docs]
        ce_scores = self.model.predict(query_doc_pairs)
        
        # Normalize cross-encoder scores to [0, 1] range
        if len(ce_scores) > 1:
            ce_min, ce_max = min(ce_scores), max(ce_scores)
            if ce_min != ce_max:
                normalized_ce_scores = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
            else:
                normalized_ce_scores = [1.0] * len(ce_scores)
        else:
            normalized_ce_scores = [1.0] * len(ce_scores)
        
        # Get reranked docs from feedback reranker
        feedback_docs = feedback_reranker.rerank_results(query, docs)
        
        # Create a mapping of doc -> position in feedback ranking
        feedback_ranks = {doc: len(docs) - i for i, doc in enumerate(feedback_docs)}
        
        # Normalize feedback ranks to [0, 1]
        max_rank = len(docs)
        normalized_feedback_scores = [feedback_ranks.get(doc, 0) / max_rank for doc in docs]
        
        # Combine scores
        combined_scores = [alpha * ce + (1-alpha) * fb 
                          for ce, fb in zip(normalized_ce_scores, normalized_feedback_scores)]
        
        # Sort by combined score
        scored_docs = list(zip(combined_scores, docs))
        reranked_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        
        return reranked_docs[:k]
    
    def hybrid_search(self, query, docs, embeddings, alpha=0.5, k=3):
        """
        Perform hybrid search by combining semantic similarity with cross-encoder scores.
        
        Args:
            query (str): The user query
            docs (list): List of document objects
            embeddings: The embedding function to use for semantic similarity
            alpha (float): Weight for semantic similarity (1-alpha for cross-encoder)
            k (int): Number of top results to return
            
        Returns:
            list: The top k documents ranked by hybrid score
        """
        self._ensure_model_loaded()
        
        if not docs:
            return []
        
        # Get cross-encoder scores
        query_doc_pairs = [[query, doc.page_content] for doc in docs]
        ce_scores = self.model.predict(query_doc_pairs)
        
        # Get semantic similarity scores
        query_embedding = embeddings.embed_query(query)
        semantic_scores = []
        
        for doc in docs:
            # Get or compute document embedding
            doc_embedding = embeddings.embed_documents([doc.page_content])[0]
            similarity = np.dot(query_embedding, doc_embedding)
            semantic_scores.append(similarity)
        
        # Normalize both score lists to [0, 1] range
        if len(ce_scores) > 1:
            ce_min, ce_max = min(ce_scores), max(ce_scores)
            if ce_min != ce_max:
                normalized_ce_scores = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
            else:
                normalized_ce_scores = [1.0] * len(ce_scores)
            
            sem_min, sem_max = min(semantic_scores), max(semantic_scores)
            if sem_min != sem_max:
                normalized_sem_scores = [(s - sem_min) / (sem_max - sem_min) for s in semantic_scores]
            else:
                normalized_sem_scores = [1.0] * len(semantic_scores)
        else:
            normalized_ce_scores = [1.0] * len(ce_scores)
            normalized_sem_scores = [1.0] * len(semantic_scores)
        
        # Combine scores
        hybrid_scores = [(1-alpha) * ce + alpha * sem 
                         for ce, sem in zip(normalized_ce_scores, normalized_sem_scores)]
        
        # Sort by hybrid score
        scored_docs = list(zip(hybrid_scores, docs))
        reranked_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        
        return reranked_docs[:k]
