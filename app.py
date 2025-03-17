import streamlit as st
import pickle
from langchain.vectorstores import VectorStore
from ui import display_ui
from utils.feedback import load_feedback_data, save_feedback
from utils.reranker import Reranker

st.set_page_config(layout="wide")

# Initialize session state if needed
if "current_chunk" not in st.session_state:
    st.session_state.current_chunk = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Load data
@st.cache_resource
def load_data():
    with open("vector_db.pkl", "rb") as f:
        vector_db = pickle.load(f)
    
    with open("reading_chunks.pkl", "rb") as f:
        reading_chunks = pickle.load(f)
    
    with open("chunk_mapping.pkl", "rb") as f:
        chunk_mapping = pickle.load(f)
    
    feedback_data = load_feedback_data()
    
    # Initialize reranker
    reranker = Reranker(vector_db)
    
    return vector_db, reading_chunks, chunk_mapping, feedback_data, reranker

# Load everything
vector_db, reading_chunks, chunk_mapping, feedback_data, reranker = load_data()

# Display the UI
display_ui(vector_db, reading_chunks, chunk_mapping, feedback_data, save_feedback, reranker)
