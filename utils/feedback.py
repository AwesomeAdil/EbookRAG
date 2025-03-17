import pickle
import os
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the feedback data file
FEEDBACK_PATH = "feedback_data.pkl"

def load_feedback_data():
    """Load feedback data from disk."""
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            # Return empty dict if there's an error loading
            return {}
    else:
        # Create a new empty feedback dict
        return {}

def get_proper_doc_id(doc, vector_db):
    """
    Get the proper document ID as stored in the vector DB.
    
    Args:
        doc: The document object from search results
        vector_db: The vector database instance
        
    Returns:
        The document ID or None if not found
    """
    # First try to get it from metadata
    if hasattr(doc, 'metadata') and 'id' in doc.metadata:
        return doc.metadata['id']
    
    # Try to get it from the document itself
    if hasattr(doc, 'id'):
        return doc.id
    
    # Try to get page_content and search for matching document in vector DB
    if hasattr(doc, 'page_content'):
        content = doc.page_content
        for doc_id, stored_doc in vector_db.docstore._dict.items():
            if stored_doc.page_content == content:
                return doc_id
    
    # If nothing works, return None
    return None

def save_feedback(query, result_index, feedback_type, vector_db):
    """
    Save user feedback on search results.
    
    Args:
        query (str): The search query
        result_index (int): The index of the result in the search results
        feedback_type (str): The type of feedback (e.g., "helpful", "not_helpful")
        vector_db: The vector database instance
        
    Returns:
        bool: True if feedback was saved successfully, False otherwise
    """
    try:
        # Load current feedback data
        feedback_data = load_feedback_data()
        
        # Get the search results from session state
        if not hasattr(st.session_state, 'search_results') or result_index >= len(st.session_state.search_results):
            logger.error(f"Error: Search result at index {result_index} not found in session state")
            return False
        
        # Get the document from search results
        doc = st.session_state.search_results[result_index]
        
        # Get the proper document ID
        doc_id = get_proper_doc_id(doc, vector_db)
        
        if doc_id is None:
            logger.error(f"Error: Could not determine document ID for result at index {result_index}")
            return False
        
        # Initialize query entry if it doesn't exist
        if query not in feedback_data:
            feedback_data[query] = {}
        
        # Save the feedback
        feedback_data[query][doc_id] = feedback_type
        
        # Write back to disk
        with open(FEEDBACK_PATH, "wb") as f:
            pickle.dump(feedback_data, f)
        
        logger.info(f"Saved '{feedback_type}' feedback for query: '{query}', doc: {doc_id}")
        return st.success("✅ Feedback saved!")
    
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False

def collect_feedback(doc_index, query, vector_db):
    """
    Generate feedback UI elements for a search result.
    
    Args:
        doc_index (int): The index of the document in search results
        query (str): The search query
        vector_db: The vector database instance
        
    Returns:
        None (displays UI elements)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Helpful", key=f"helpful_{doc_index}"):
            if save_feedback(query, doc_index, "helpful", vector_db):
                st.success("Feedback saved as helpful!")
            else:
                st.error(f"Error: Failed to save feedback")
    
    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{doc_index}"):
            if save_feedback(query, doc_index, "not_helpful", vector_db):
                st.success("Feedback saved as not helpful!")
            else:
                st.error(f"Error: Failed to save feedback")
