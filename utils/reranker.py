import pickle
import numpy as np
import datetime
import os
from sklearn.ensemble import RandomForestClassifier

class Reranker:
    """A reranking model that improves search results based on user feedback."""
    
    def __init__(self, vector_db, model_path="reranker_model.pkl", min_feedback_entries=10):
        self.vector_db = vector_db
        self.model_path = model_path
        self.min_feedback_entries = min_feedback_entries
        
        # Load model if it exists
        self.model = self._load_model()
        self.last_training_time = self._get_last_training_time()
    
    def rerank_results(self, query, docs, k=3):
        """Rerank search results based on the trained model."""
        # If no model exists, return original results
        if self.model is None:
            return docs[:k]
        
        # Extract features for each document
        features = []
        for doc in docs:
            feature_vector = self._extract_features(query, doc)
            features.append(feature_vector)
        
        if not features:
            return docs[:k]
        
        # Get relevance predictions
        relevance_scores = self.model.predict_proba(features)
        
        # Sort by relevance score
        doc_scores = [(doc, score[1]) for doc, score in zip(docs, relevance_scores)]
        reranked_docs = [doc for doc, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]
        
        return reranked_docs[:k]
    
    def train_model(self, force=False):
        """Train the reranking model based on feedback data."""
        # Load feedback data
        with open("feedback_data.pkl", "rb") as f:
            feedback_data = pickle.load(f)
        
        # Check if we have enough data
        total_feedback = sum(len(results) for results in feedback_data.values())
        if total_feedback < self.min_feedback_entries and not force:
            return False, f"Not enough feedback data ({total_feedback}/{self.min_feedback_entries})"
        
        # Prepare training data
        X, y = [], []
        
        for query, results in feedback_data.items():
            for doc_id, feedback in results.items():
                # Skip if doc_id is not in the docstore
                if doc_id not in self.vector_db.docstore._dict:
                    continue
                
                # Get document
                doc = self.vector_db.docstore._dict[doc_id]
                
                # Extract features
                features = self._extract_features(query, doc)
                
                # Add to training data
                X.append(features)
                y.append(1 if feedback == "helpful" else 0)
        
        # Train model
        if len(X) > 0:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save model
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Update model and training time
            self.model = model
            self._update_training_time()
            
            return True, f"Model trained successfully with {len(X)} examples"
        
        return False, "No valid training examples found"
    
    def should_retrain(self):
        """Check if the model should be retrained."""
        # If no model exists, train it
        if self.model is None:
            return True
        
        # Check if it's been at least a week since last training
        if self.last_training_time:
            one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
            if self.last_training_time > one_week_ago:
                return False
        
        # Check if we have enough feedback data
        with open("feedback_data.pkl", "rb") as f:
            feedback_data = pickle.load(f)
        
        total_feedback = sum(len(results) for results in feedback_data.values())
        return total_feedback >= self.min_feedback_entries
    
    def _extract_features(self, query, doc):
        """Extract features for a query-document pair."""
        # Get embeddings
        query_embedding = self.vector_db.embedding_function.embed_query(query)
        doc_embedding = self.vector_db.embedding_function.embed_documents([doc.page_content])[0]
        
        # Calculate similarity
        similarity = np.dot(query_embedding, doc_embedding)
        
        # Get text length features
        doc_length = len(doc.page_content.split())
        query_length = len(query.split())
        
        # Calculate word overlap
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        word_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
        
        # Return feature vector
        return [similarity, doc_length, query_length, word_overlap]
    
    def _load_model(self):
        """Load model from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _get_last_training_time(self):
        """Get the last time the model was trained."""
        if os.path.exists(self.model_path):
            return datetime.datetime.fromtimestamp(os.path.getmtime(self.model_path))
        return None
    
    def _update_training_time(self):
        """Update the last training time."""
        self.last_training_time = datetime.datetime.now()
