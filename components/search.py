import streamlit as st
from reranking_utils import get_reranked_results

def display_search(vector_db, reading_chunks, feedback_data, save_feedback, feedback_reranker):
    """Display search functionality to query the eBook."""
    
    st.subheader("🔍 Ask about the book")
    query = st.text_input("Enter your question:")
    
    # Add reranking strategy selector
    reranking_options = {
        "hybrid": "Hybrid (Semantic + Cross-Encoder)",
        "ensemble": "Ensemble (Cross-Encoder + Feedback)",
        "cross_encoder": "Cross-Encoder Only",
        "feedback": "Feedback-Based Only",
        "semantic": "Semantic Search Only"
    }
    
    with st.expander("Search Settings", expanded=False):
        reranking_strategy = st.selectbox(
            "Reranking Strategy",
            options=list(reranking_options.keys()),
            format_func=lambda x: reranking_options[x],
            index=0
        )
        
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
        initial_results = st.slider("Initial retrieval count", min_value=5, max_value=30, value=10)
    
    if query:
        try:
            with st.spinner("Searching..."):
                # Get reranked results using the utility function
                docs = get_reranked_results(
                    query=query,
                    vector_db=vector_db,
                    feedback_reranker=feedback_reranker,
                    reranking_strategy=reranking_strategy,
                    k=num_results,
                    initial_k=initial_results
                )
                
                st.session_state.search_results = docs
            
            # Display results
            for i, doc in enumerate(docs):
                with st.expander(f"Result {i+1}", expanded=True):
                    st.markdown(doc.page_content)
                    
                    # Feedback controls
                    feedback_cols = st.columns([1, 1, 1])
                    with feedback_cols[0]:
                        if st.button("👍 Helpful", key=f"helpful_{i}"):
                            print(save_feedback(query, i, "helpful", vector_db))
                    
                    with feedback_cols[1]:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                            print(save_feedback(query, i, "not_helpful", vector_db))
                    
                    # Jump to reading chunk button
                    with feedback_cols[2]:
                        read_idx = get_reading_index(doc, reading_chunks)
                        
                        if st.button(f"📖 Read", key=f"goto_{i}"):
                            st.session_state.current_chunk = read_idx
                            # Add to history
                            if len(st.session_state.history) == 0 or st.session_state.history[-1] != st.session_state.current_chunk:
                                st.session_state.history.append(st.session_state.current_chunk)
                            st.rerun()
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

def get_reading_index(doc, reading_chunks):
    """Determine the reading index for a document."""
    # Try to get the reading index from metadata
    if "read_index" in doc.metadata:
        return doc.metadata["read_index"]
    elif "index" in doc.metadata:  # Fallback to regular index
        return doc.metadata["index"]
    else:
        # Try to find a matching chunk
        for idx, chunk in enumerate(reading_chunks):
            if doc.page_content in chunk:
                return idx
        return 0  # Default to first chunk if no match found
