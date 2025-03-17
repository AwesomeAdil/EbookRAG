import streamlit as st
import pickle
import os
from typing import Dict, Any, List, Optional
from components.search import get_reading_index
from reranking_utils import get_reranked_results

# Import the LLM-related components
from book_summary_generator import load_book_summary
from llm_context_handler import LLMContextHandler

def load_llm_components(
    reading_chunks_path: str = "reading_chunks.pkl",
    vector_db_path: str = "vector_db.pkl",
    book_summary_path: str = "book_summary.pkl"
) -> tuple:
    """Load necessary components for LLM functionality."""
    # Load book summary if it exists
    book_summary = None
    if os.path.exists(book_summary_path):
        book_summary = load_book_summary(book_summary_path)
    
    # Initialize context handler
    context_handler = LLMContextHandler(
        reading_chunks_path=reading_chunks_path,
        vector_db_path=vector_db_path,
        book_summary=book_summary
    )
    
    return book_summary, context_handler

def format_llm_query(
    query: str,
    context_handler: LLMContextHandler,
    search_results: List,
    reranking_info: Optional[Dict] = None,
    include_summary: bool = True
) -> str:
    """Format a query for the LLM with context from search results."""
    # Prepare context package
    context_package = context_handler.prepare_query_response(
        query=query,
        search_results=search_results,
        reranking_info=reranking_info,
        include_summary=include_summary
    )
    
    # Return the formatted context
    return context_package["formatted_context"]

def display_question_answering(
    vector_db, 
    reading_chunks, 
    feedback_data, 
    save_feedback, 
    reranker,
    context_handler,
    book_summary,
    llm_type: str = "placeholder"
):
    """Display the question answering interface."""
    st.subheader("🧠 Ask about the Book")
    query = st.text_input("Enter your question about the book:")
    
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
            "Reranking Strategy LLM",
            options=list(reranking_options.keys()),
            format_func=lambda x: reranking_options[x],
            index=0
        )
        
        num_results = st.slider("Number of results LLM", min_value=1, max_value=10, value=3)
        initial_results = st.slider("Initial retrieval count LLM", min_value=5, max_value=30, value=10)
        
        # Context settings
        include_summary = st.checkbox("Include book summary in context", value=True)
        
        # LLM type placeholder
        st.markdown(f"**LLM Type:** {llm_type} (connect your preferred LLM here)")
    
    if query:
        try:
            with st.spinner("Searching and preparing context..."):
                # Get reranked results
                docs = get_reranked_results(
                    query=query,
                    vector_db=vector_db,
                    feedback_reranker=reranker,
                    reranking_strategy=reranking_strategy,
                    k=num_results,
                    initial_k=initial_results
                )
                
                # Save search results
                st.session_state.search_results = docs
                
                # Prepare reranking info
                reranking_info = {
                    "strategy": reranking_strategy,
                    "scores": [1.0] * len(docs)  # Placeholder scores
                }
                
                # Format context for LLM
                llm_ready_context = format_llm_query(
                    query=query,
                    context_handler=context_handler,
                    search_results=docs,
                    reranking_info=reranking_info,
                    include_summary=include_summary
                )
                
                # Save context
                st.session_state.llm_context = llm_ready_context
            
            # Display results and context
            with st.expander("LLM-Ready Context", expanded=False):
                st.code(llm_ready_context, language="markdown")
                
                st.info("""
                This is the formatted context that would be sent to your LLM.
                Connect your preferred LLM here to get answers to your questions.
                """)
            
            # Display search results
            st.subheader("Relevant Passages")
            for i, doc in enumerate(docs):
                with st.expander(f"Passage {i+1}", expanded=(i == 0)):
                    st.markdown(doc.page_content)
                    
                    # Feedback and reading controls
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        if st.button("👍 Helpful", key=f"helpful_{i}"):
                            save_feedback(query, i, "helpful", vector_db)
                    
                    with cols[1]:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                            save_feedback(query, i, "not_helpful", vector_db)
                    
                    # Jump to reading chunk button
                    with cols[2]:
                        read_idx = get_reading_index(doc, reading_chunks)
                        
                        if st.button(f"📖 Read", key=f"goto_{i}"):
                            st.session_state.current_chunk = read_idx
                            # Add to history
                            if len(st.session_state.history) == 0 or st.session_state.history[-1] != st.session_state.current_chunk:
                                st.session_state.history.append(st.session_state.current_chunk)
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
