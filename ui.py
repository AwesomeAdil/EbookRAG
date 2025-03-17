import streamlit as st
from components.reader import display_reader
from components.search import display_search
from components.history import display_history
from components.settings import display_settings
from components.question_answering import display_qa_component

def display_ui(vector_db, reading_chunks, chunk_mapping, feedback_data, save_feedback, reranker):
    """Main UI layout for the eBook reader application."""
    
    st.title("📖 AI-Powered eBook Reader & Query Assistant")

    # Put settings in the sidebar
    with st.sidebar:
        st.header("Settings")
        display_settings(vector_db, reading_chunks, chunk_mapping, feedback_data, reranker)

    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["📚 Reader", "🧠 Ask Questions"])
    
    # Reader tab
    with tab1:
        # Create a two-column layout with more space
        col1, col2 = st.columns([3, 1])

        # Left column: eBook Reader
        with col1:
            display_reader(reading_chunks)

        # Right column: Search, History
        with col2:
            # Search component
            display_search(vector_db, reading_chunks, feedback_data, save_feedback, reranker)
            
            # History component
            display_history(reading_chunks)
    
    # Ask Questions tab
    with tab2:
        display_qa_component(vector_db, reading_chunks, feedback_data, save_feedback, reranker)

    # Footer
    st.markdown("---")
    st.markdown("*AI-Powered eBook Reader - For educational purposes only*")
