import streamlit as st
from llm_integration import display_question_answering, load_llm_components

def display_qa_component(vector_db, reading_chunks, feedback_data, save_feedback, reranker):
    """Wrapper for displaying the question answering component."""
    
    # Initialize LLM components if not already done
    if "book_summary" not in st.session_state or "context_handler" not in st.session_state:
        book_summary, context_handler = load_llm_components()
        st.session_state.book_summary = book_summary
        st.session_state.context_handler = context_handler
    
    # Display the question answering interface
    display_question_answering(
        vector_db=vector_db,
        reading_chunks=reading_chunks,
        feedback_data=feedback_data,
        save_feedback=save_feedback,
        reranker=reranker,
        context_handler=st.session_state.context_handler,
        book_summary=st.session_state.book_summary,
        llm_type="Connect your LLM here"
    )
