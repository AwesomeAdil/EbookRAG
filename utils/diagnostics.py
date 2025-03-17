import pandas as pd
from typing import Dict, List, Tuple, Any

def run_diagnostics(vector_db, feedback_data, reranker):
    """
    Run comprehensive diagnostics on the system components.
    
    Args:
        vector_db: The vector database instance
        feedback_data: The current feedback data dictionary
        reranker: The reranker model instance
        
    Returns:
        dict: Dictionary containing diagnostic results
    """
    results = {}
    
    # Vector DB stats
    doc_count = len(vector_db.docstore._dict) if hasattr(vector_db.docstore, '_dict') else 0
    results['doc_count'] = doc_count
    
    # Get sample document IDs from vector DB
    sample_size = min(5, doc_count)
    if doc_count > 0:
        sample_ids = list(vector_db.docstore._dict.keys())[:sample_size]
        results['sample_ids'] = sample_ids
    else:
        results['sample_ids'] = []
    
    # Extract all document IDs from vector DB
    vector_ids = list(vector_db.docstore._dict.keys()) if hasattr(vector_db.docstore, '_dict') else []
    results['vector_ids'] = vector_ids
    results['vector_id_type'] = type(vector_ids[0]).__name__ if vector_ids else None
    
    # Extract all document IDs from feedback data
    feedback_ids = []
    for query, docs in feedback_data.items():
        feedback_ids.extend(list(docs.keys()))
    
    results['feedback_ids'] = feedback_ids
    results['feedback_id_type'] = type(feedback_ids[0]).__name__ if feedback_ids else None
    
    # Check for consistency between feedback data and vector DB
    missing_docs = []
    for query, docs in feedback_data.items():
        for doc_id in docs.keys():
            if doc_id not in vector_db.docstore._dict:
                missing_docs.append((query, doc_id))
    
    results['missing_docs'] = missing_docs
    
    return results

def fix_feedback_data(feedback_data, vector_db, strategy="map"):
    """
    Fix inconsistencies in feedback data.
    
    Args:
        feedback_data: The feedback data dictionary to fix
        vector_db: The vector database to check against
        strategy: Either "map" to map IDs or "delete" to remove invalid entries
    
    Returns:
        int: Number of fixes applied
    """
    if not feedback_data:
        return 0
    
    fix_count = 0
    queries_to_update = {}
    
    # Collect all fixes needed
    for query, docs in feedback_data.items():
        invalid_doc_ids = []
        
        for doc_id in list(docs.keys()):
            if doc_id not in vector_db.docstore._dict:
                invalid_doc_ids.append(doc_id)
        
        if invalid_doc_ids:
            queries_to_update[query] = invalid_doc_ids
    
    # Apply fixes
    if strategy == "delete":
        # Remove invalid entries
        for query, invalid_ids in queries_to_update.items():
            for doc_id in invalid_ids:
                if doc_id in feedback_data[query]:
                    del feedback_data[query][doc_id]
                    fix_count += 1
            
            # Remove the query if no documents left
            if len(feedback_data[query]) == 0:
                del feedback_data[query]
    
    elif strategy == "map":
        # Try to map to actual IDs (more complex)
        # This is a simplified implementation that would need to be enhanced
        # based on your specific ID format and document matching logic
        for query, invalid_ids in queries_to_update.items():
            for invalid_id in invalid_ids:
                # For string IDs, try to find a close match
                if isinstance(invalid_id, str):
                    for valid_id in vector_db.docstore._dict.keys():
                        if isinstance(valid_id, str) and invalid_id in valid_id or valid_id in invalid_id:
                            # Found a potential match, move the feedback
                            feedback_data[query][valid_id] = feedback_data[query][invalid_id]
                            del feedback_data[query][invalid_id]
                            fix_count += 1
                            break
                
                # If we couldn't map, delete as a fallback
                if invalid_id in feedback_data[query]:
                    del feedback_data[query][invalid_id]
                    fix_count += 1
            
            # Remove the query if no documents left
            if len(feedback_data[query]) == 0:
                del feedback_data[query]
    
    return fix_count
